import time
import pandas as pd
import math
import argparse
import queue
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import StaleElementReferenceException, ElementClickInterceptedException
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import tempfile
import random
import socket
# shutil back for profile directory copying
import os
import shutil
# shutil and threading removed; not used currently

# 1. 데이터 로드 (테스트용 데이터 프레임 생성)
# 실제 사용 시에는 pd.read_csv 사용
df = pd.read_csv('아파트_수집_최종.csv')
# data = {
#     'kaptCode': ['A10023990'],
#     'kaptName': ['청년주택 와이엔타워'],
#     'doroJuso': ['서울특별시 노원구 공릉동 동일로 1000']
# }
# df = pd.DataFrame(data)

# 2. 브라우저 옵션 강화 (봇 탐지 회피 및 안정성 확보)
chrome_options = Options()

TEMP_DATA_PATH = os.path.join(os.getcwd(), "temp_profile") 

# 프로필 폴더가 없으면 생성
if not os.path.exists(TEMP_DATA_PATH):
    os.makedirs(TEMP_DATA_PATH)
    print(f"새 임시 프로필 폴더 생성: {TEMP_DATA_PATH}")

chrome_options.add_argument(f"--user-data-dir={TEMP_DATA_PATH}")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080") # 화면 크기 고정 (반응형 레이아웃 문제 방지)
chrome_options.add_argument("lang=ko_KR")

# 봇 탐지 방지용 핵심 옵션
chrome_options.add_argument("--disable-blink-features=AutomationControlled") 
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)

# User-Agent 변경
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

def create_driver(headless=False, worker_id=None, base_profile_dir=TEMP_DATA_PATH, clone_profile=False, cleanup_clone=False, force_new_profile=False):
    # Create a unique profile folder per worker to avoid profile lock conflicts
    if clone_profile and worker_id is not None and not force_new_profile:
        profile_dir = os.path.join(base_profile_dir, f"worker_{worker_id}")
        # If base profile exists and clone doesn't exist, create a copy
        if os.path.exists(base_profile_dir) and not os.path.exists(profile_dir):
            try:
                print(f"Cloning profile from {base_profile_dir} -> {profile_dir}")
                shutil.copytree(base_profile_dir, profile_dir)
            except Exception as e:
                print(f"Failed to clone profile, creating fresh profile {profile_dir}: {e}")
                os.makedirs(profile_dir, exist_ok=True)
        elif not os.path.exists(profile_dir):
            os.makedirs(profile_dir, exist_ok=True)
    else:
        profile_dir = base_profile_dir if worker_id is None else os.path.join(base_profile_dir, f"worker_{worker_id}")
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir, exist_ok=True)

    # Make sure profile_dir path is absolute
    profile_dir = os.path.abspath(profile_dir)

    # Validate we can write into profile_dir; if not, fallback to a temp dir
    try:
        test_file = os.path.join(profile_dir, ".profile_test_write")
        with open(test_file, 'w') as f:
            f.write('ok')
        os.remove(test_file)
    except Exception as e:
        print(f"Warning: cannot write to profile directory '{profile_dir}': {e}")
        # Fallback to a unique temporary directory
        profile_dir = tempfile.mkdtemp(prefix='zippoom_profile_')
        print(f"Using fallback profile directory: {profile_dir}")

    # alter chrome options per worker
    options = Options()
    options.add_argument(f"--user-data-dir={profile_dir}")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("lang=ko_KR")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    if headless:
        # Use new headless mode for modern Chrome; fallback to legacy if needed
        try:
            options.add_argument("--headless=new")
        except Exception:
            options.add_argument("--headless")

    # Assign a unique remote-debugging-port to reduce collisions on Windows
    try:
        port = random.randint(20000, 40000)
        # verify port is free
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                s.listen(1)
                options.add_argument(f"--remote-debugging-port={port}")
            except Exception:
                # If port bind fails, skip adding and let ChromeDriver manage
                pass
    except Exception:
        pass

    # Try starting Chrome with retries; on failure make a fresh profile and try once more
    drv = None
    for attempt in range(2):
        try:
            drv = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            print(f"✅ 드라이버 세션 생성 성공 (worker {worker_id})")
            break
        except Exception as e:
            print(f"❌ 드라이버 세션 생성 실패 (worker {worker_id}) (attempt {attempt+1}): {e}")
            # If cloning was used, try creating a fresh profile and retry once
            if clone_profile and attempt == 0 and worker_id is not None:
                try:
                    if os.path.exists(profile_dir):
                        shutil.rmtree(profile_dir)
                    os.makedirs(profile_dir, exist_ok=True)
                    print(f"Attempting again with fresh profile {profile_dir}")
                    continue
                except Exception as ee:
                    print(f"Failed to reset profile for retry: {ee}")
            # No more retries, re-raise
            raise

    if drv is None:
        raise Exception(f"Failed to start Chrome for worker {worker_id}")

    # Hide webdriver property to reduce detection
    try:
        drv.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })
    except Exception:
        pass
    input()
    return drv

def save_cookies(driver, cookie_file):
    try:
        cookies = driver.get_cookies()
        with open(cookie_file, 'w', encoding='utf-8') as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
        print(f"Cookies saved to {cookie_file}")
    except Exception as e:
        print(f"Failed to save cookies: {e}")


def load_cookies(driver, cookie_file, url=None):
    try:
        with open(cookie_file, 'r', encoding='utf-8') as f:
            cookies = json.load(f)
    except Exception as e:
        print(f"Failed to load cookies from {cookie_file}: {e}")
        return False

    if url:
        driver.get(url)

    for c in cookies:
        # Selenium add_cookie expects name and value at minimum, other fields optional
        cookie = {k: v for k, v in c.items() if k != 'sameSite'}
        try:
            # remove 'expiry' if not int
            if 'expiry' in cookie:
                try:
                    cookie['expiry'] = int(cookie['expiry'])
                except Exception:
                    cookie.pop('expiry', None)
            driver.add_cookie(cookie)
        except Exception as e:
            # ignore failures for cookies that can't be added
            print(f"Warning: add_cookie failed for {cookie.get('name')}: {e}")
    try:
        driver.refresh()
    except Exception:
        pass
    return True


def is_logged_in(driver):
    """Basic heuristic to check login status. Adjust for Zippoom as needed."""
    try:
        # Look for '로그아웃' or profile avatar
        logout_xpaths = [
            "//button[contains(., '로그아웃') or contains(., '로그아웃하기')]",
            "//a[contains(@href, 'logout') or contains(., '로그아웃')]",
            "//div[contains(@class, 'avatar') or contains(@class, 'profile')]"
        ]
        for xp in logout_xpaths:
            els = driver.find_elements(By.XPATH, xp)
            if els:
                return True
        # If there's a login text visible, not logged in
        login_xps = ["//button[contains(., '로그인')]", "//a[contains(., '로그인')]"]
        for xp in login_xps:
            els = driver.find_elements(By.XPATH, xp)
            if els:
                return False
    except Exception:
        pass
    # fallback: assume not logged in
    return False


def worker_thread(worker_id, rows, results_q, headless=False, base_profile_dir=TEMP_DATA_PATH, cookies_file=None, reuse_cookies=False, clone_profile=False, cleanup_clone=False, force_new_profile=False):
    driver = None
    try:
        driver = create_driver(headless=headless, worker_id=worker_id, base_profile_dir=base_profile_dir, clone_profile=clone_profile, cleanup_clone=cleanup_clone, force_new_profile=force_new_profile)
        # Stagger start so that all workers don't hit the server at once
        try:
            stagger = 0.5 * (worker_id or 0)
        except Exception:
            stagger = 0
        if stagger:
            time.sleep(stagger)
        # Load cookies if requested
        if reuse_cookies and cookies_file and os.path.exists(cookies_file):
            print(f"Worker {worker_id} - Loading cookies from {cookies_file}")
            # Ensure we have a base site open so cookies can be added
            try:
                load_cookies(driver, cookies_file, url='https://zippoom.com/')
            except Exception as e:
                print(f"Worker {worker_id} - load_cookies error: {e}")
            try:
                if is_logged_in(driver):
                    print(f"Worker {worker_id} - login session loaded successfully")
                else:
                    print(f"Worker {worker_id} - Not logged in after applying cookies")
            except Exception:
                print(f"Worker {worker_id} - Could not confirm login state")
        for idx, row in rows.iterrows():
            try:
                reviews = crawl_zippoom(row['doroJuso'], row['kaptName'], driver)
                for r in reviews:
                    results_q.put(r)
                # polite pause between rows
                time.sleep(1)
            except Exception as e:
                print(f"Worker {worker_id} - Error processing row {idx}: {e}")
                continue
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
        # optionally cleanup cloned profile after finished
        if clone_profile and cleanup_clone and worker_id is not None:
            clone_dir = os.path.join(base_profile_dir, f"worker_{worker_id}")
            if os.path.exists(clone_dir):
                try:
                    shutil.rmtree(clone_dir)
                    print(f"Cleaned up profile {clone_dir}")
                except Exception as e:
                    print(f"Failed to cleanup profile {clone_dir}: {e}")

def print_progress(total, q):
    # no-op helper to format progress if needed
    print(f"Collected so far: {q.qsize()}/{total}")


def crawl_zippoom(doro_juso, kapt_name, driver):
    collected_reviews = []
    
    # 1. 검색 전략 수립
    search_candidates = []
    is_valid_juso = False
    if doro_juso is not None:
        if isinstance(doro_juso, float):
            if not math.isnan(doro_juso): is_valid_juso = True
        elif str(doro_juso).strip().lower() != 'nan' and str(doro_juso).strip() != "":
            is_valid_juso = True

    if is_valid_juso: search_candidates.append((doro_juso, "도로명 주소"))
    search_candidates.append((kapt_name, "아파트 이름"))
    
    print(f"\n🔍 크롤링 시작 대상: {kapt_name}")

    success_search = False
    
    for keyword, desc in search_candidates:
        print(f"  🔄 전략 시도: '{desc}'로 검색 ({keyword})")
        
        try:
            # Use the search page directly so we can type into the visible search field
            driver.get("https://zippoom.com/search")
            # 페이지 로딩 후 React가 안정을 찾을 때까지 조금 넉넉히 대기
            time.sleep(2) 
            
            wait = WebDriverWait(driver, 10)
            
            # =========================================================
            # [Step 1] 검색창 선택 및 입력 (검색 페이지에서 직접 키 입력)
            # We load the /search page, locate visible input field and type using send_keys
            # =========================================================
            input_success = False
            attempts = 0
            # Prefer the specific input with enterkeyhint='search' (site's search box)
            input_selectors = [
                # exact target per user's request: clickable input with the class
                "//input[contains(@class, 'absolute') and contains(@class, 'z-20') and @enterkeyhint='search']",
                "//input[@enterkeyhint='search']",
                # fallback: search type or placeholder/class matches
                "//input[@type='search']",
                "//input[@type='text' and contains(@placeholder, '검색')]",
                "//input[@type='text' and contains(@placeholder, '주소')]",
                "//input[@type='text' and contains(@placeholder, '건물명')]",
                "//input[contains(@class, 'search') or contains(@class, 'Search') or contains(@class, 'searchInput')]",
                "//input[contains(@placeholder, '검색') or contains(@placeholder, '주소') or contains(@placeholder, '건물명')]",
                "//input[@role='searchbox']",
            ]

            while not input_success and attempts < 3:
                try:
                    attempts += 1
                    real_input = None
                    for sel in input_selectors:
                        try:
                            real_input = wait.until(EC.element_to_be_clickable((By.XPATH, sel)))
                            if real_input:
                                break
                        except Exception:
                            # ignore and try next selector
                            continue

                    if not real_input:
                        raise Exception("검색 input을 찾을 수 없습니다.")

                    # Use ActionChains to move to the element, click, and type using keyboard
                    ac = ActionChains(driver)
                    ac.move_to_element(real_input).click().send_keys(Keys.CONTROL + "a").send_keys(Keys.BACKSPACE).send_keys(keyword).pause(0.2).send_keys(Keys.RETURN).perform()

                    input_success = True
                    print(f"  👉 [Step 1] 입력 성공 (시도 {attempts}회)")

                except (StaleElementReferenceException, ElementClickInterceptedException):
                    print(f"  ⚠️ 요소가 변경됨(Stale). 재시도 중... ({attempts}/3)")
                    time.sleep(1)
                except Exception as e:
                    print(f"  ⚠️ 입력 중 일반 에러: {e}")
                    break

            if not input_success:
                print("  ❌ 검색어 입력 실패. 다음 전략으로.")
                continue

            # =========================================================
            # [Step 2] 결과 확인 및 클릭
            # =========================================================
            time.sleep(3)
            xpath_result = "//button[.//span[contains(text(), '도로명')]]"
            first_result = wait.until(EC.element_to_be_clickable((By.XPATH, xpath_result)))
            
            driver.execute_script("arguments[0].click();", first_result)
            print(f"  ✅ 검색 성공! 상세 페이지로 이동합니다.")
            success_search = True
            time.sleep(3)
            break 
            
        except Exception as e:
            print(f"  ⚠️ 실패: {desc} 검색 결과 없음 ({e})")
            continue 

    if not success_search:
        print("  ❌ 모든 검색 전략 실패. 다음 아파트로 넘어갑니다.")
        return []

    # =========================================================
    # [Step 3] 리뷰 탭 클릭
    # =========================================================
    try:
        wait = WebDriverWait(driver, 5)
        xpath_tab = "//p[contains(@class, 'cursor-pointer') and contains(., '리뷰')]"
        review_tab = wait.until(EC.element_to_be_clickable((By.XPATH, xpath_tab)))
        driver.execute_script("arguments[0].click();", review_tab)
        print("  👉 리뷰 탭 클릭 성공")
        time.sleep(2)
    except:
        print("  ℹ️ 리뷰 탭 클릭 건너뜀")

    # =========================================================
    # [Step 4] '더보기' 반복 클릭
    # =========================================================
    print("  🔄 리뷰 전체 로딩 중...")
    while True:
        try:
            more_btn = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., '거주 후기 더보기')]"))
            )
            driver.execute_script("arguments[0].click();", more_btn)
            time.sleep(1)
        except:
            break 

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)

    # =========================================================
    # [Step 5] 데이터 추출
    # =========================================================
    review_blocks = driver.find_elements(By.XPATH, "//div[@data-testid='리뷰']")
    
    for block in review_blocks:
        review_item = {'kaptName': kapt_name, 'doroJuso': doro_juso, 'Score': None, 'Pros': None, 'Cons': None}
        
        try:
            full_btn = block.find_element(By.XPATH, ".//p[contains(text(), '전체 보기')]/..")
            driver.execute_script("arguments[0].click();", full_btn)
            time.sleep(0.1)
        except: pass

        try: review_item['Score'] = block.find_element(By.XPATH, ".//p[contains(@class, 'font-bold')]").text
        except: pass
        try: review_item['Pros'] = block.find_element(By.XPATH, ".//p[text()='장점']/following-sibling::p[1]").text
        except: review_item['Pros'] = ""
        try: review_item['Cons'] = block.find_element(By.XPATH, ".//p[text()='단점']/following-sibling::p[1]").text
        except: review_item['Cons'] = ""
        
        collected_reviews.append(review_item)

    print(f"  🎉 수집 완료: {len(collected_reviews)}건")
    return collected_reviews
    
# driver 인스턴스와 crawl_zippoom 함수는 정의되어 있다고 가정합니다.

# 결과를 담을 리스트 (defaultdict를 사용하는 것이 효율적이지만, 
# 기존처럼 list of dicts로 유지하여 Pandas로 변환합니다.)
def split_dataframe(df, n):
    """Split a dataframe into n nearly-equal parts."""
    if n <= 1:
        return [df]
    k, m = divmod(len(df), n)
    parts = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < m else 0)
        parts.append(df.iloc[start:end])
        start = end
    return parts


def main():
    parser = argparse.ArgumentParser(description="Zippoom review crawler with multi-threading and headless options")
    parser.add_argument('--headless', action='store_true', help='Run Chrome in headless mode')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads to run')
    parser.add_argument('--save', type=str, default='리뷰_구조화_결과.csv', help='Output CSV file')
    parser.add_argument('--cookies-file', type=str, default='cookies.json', help='Path to cookies file to save/load')
    parser.add_argument('--record-cookies', action='store_true', help='Open browser for manual login and save cookies to --cookies-file')
    parser.add_argument('--reuse-cookies', action='store_true', help='Load cookies from --cookies-file for each worker')
    parser.add_argument('--profile-dir', type=str, default=TEMP_DATA_PATH, help='Base profile directory to reuse or clone')
    parser.add_argument('--force-new-profile', action='store_true', help='Force using fresh profile per worker; skip cloning from base profile')
    parser.add_argument('--clone-profile', action='store_true', help='Clone the profile-dir per worker to preserve login data without file-lock conflicts')
    parser.add_argument('--cleanup-cloned-profiles', action='store_true', help='Remove cloned profile directories when worker finishes')
    args = parser.parse_args()

    total_rows = len(df)
    results_q = queue.Queue()

    # split work
    workers = max(1, args.workers)
    parts = split_dataframe(df, workers)

    # If asked to record cookies, open a non-headless browser for manual login and save cookies
    if args.record_cookies:
        print("Please log in on the opened browser window; after login, press Enter in this console to save cookies.")
        # Use the specified base profile dir (non-worker) so login remains in that profile
        login_drv = create_driver(headless=False, worker_id=None, base_profile_dir=args.profile_dir)
        try:
            login_drv.get('https://zippoom.com/')
            input('After logging in, press Enter to save cookies...')
            save_cookies(login_drv, args.cookies_file)
        finally:
            try:
                login_drv.quit()
            except Exception:
                pass

    # Run workers
    if workers == 1:
        print("Running in single-threaded mode")
        # For single worker, use base profile directly (worker_id=None) so profile-dir maps to the provided profile
        worker_thread(None, parts[0], results_q, headless=args.headless, base_profile_dir=args.profile_dir, cookies_file=args.cookies_file, reuse_cookies=args.reuse_cookies, clone_profile=args.clone_profile, cleanup_clone=args.cleanup_cloned_profiles, force_new_profile=args.force_new_profile)
    else:
        print(f"Running with {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(worker_thread, wid, parts[wid], results_q,
                                 headless=args.headless,
                                 base_profile_dir=args.profile_dir,
                                 cookies_file=args.cookies_file,
                                 reuse_cookies=args.reuse_cookies,
                                 clone_profile=args.clone_profile,
                                 cleanup_clone=args.cleanup_cloned_profiles,
                                 force_new_profile=args.force_new_profile) for wid in range(workers)]
            # Wait for all to complete
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"Worker thrown an exception: {e}")

    # Collect results from queue
    reviews_data = []
    while not results_q.empty():
        reviews_data.append(results_q.get())

    print(f"\n총 수집된 리뷰: {len(reviews_data)}건")

    if reviews_data:
        df_final_reviews = pd.DataFrame(reviews_data)
        df_final_reviews.to_csv(args.save, index=False, encoding='utf-8-sig')
        print(f"✅ 리뷰 데이터가 '{args.save}'로 저장되었습니다.")


if __name__ == '__main__':
    main()