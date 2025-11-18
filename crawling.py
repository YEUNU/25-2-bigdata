import time
import pandas as pd
import math
import argparse
import json
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


def get_processed_indices(csv_path):
    """CSV 파일에서 이미 처리된 인덱스 목록을 반환"""
    if not os.path.exists(csv_path):
        return set()
    try:
        df_existing = pd.read_csv(csv_path)
        if 'source_index' in df_existing.columns:
            return set(df_existing['source_index'].unique())
        return set()
    except Exception as e:
        print(f"Warning: CSV 파일 읽기 실패: {e}")
        return set()


def append_to_csv(reviews_data, csv_path):
    """리뷰 데이터를 CSV에 추가 저장"""
    if not reviews_data:
        return
    
    df_new = pd.DataFrame(reviews_data)
    
    # 파일이 없거나 비어있으면 헤더 포함해서 생성
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        df_new.to_csv(csv_path, index=False, encoding='utf-8-sig')
    else:
        # 파일이 있으면 헤더 없이 추가
        df_new.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')


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
            time.sleep(2 + random.uniform(0, 0.3)) 
            
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
                    time.sleep(1 + random.uniform(0, 0.3))
                except Exception as e:
                    print(f"  ⚠️ 입력 중 일반 에러: {e}")
                    break

            if not input_success:
                print("  ❌ 검색어 입력 실패. 다음 전략으로.")
                continue

            # =========================================================
            # [Step 2] 결과 확인 및 클릭
            # =========================================================
            time.sleep(3 + random.uniform(0, 0.3))
            xpath_result = "//button[.//span[contains(text(), '도로명')]]"
            first_result = wait.until(EC.element_to_be_clickable((By.XPATH, xpath_result)))
            
            driver.execute_script("arguments[0].click();", first_result)
            print(f"  ✅ 검색 성공! 상세 페이지로 이동합니다.")
            success_search = True
            time.sleep(3 + random.uniform(0, 0.3))
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
        time.sleep(2 + random.uniform(0, 0.3))
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
            time.sleep(1 + random.uniform(0, 0.3))
        except:
            break 

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1 + random.uniform(0, 0.3))

    # =========================================================
    # [Step 5] 데이터 추출
    # =========================================================
    review_blocks = driver.find_elements(By.XPATH, "//div[@data-testid='리뷰']")
    
    for block in review_blocks:
        review_item = {'kaptName': kapt_name, 'doroJuso': doro_juso, 'Score': None, 'Pros': None, 'Cons': None}
        
        try:
            full_btn = block.find_element(By.XPATH, ".//p[contains(text(), '전체 보기')]/..")
            driver.execute_script("arguments[0].click();", full_btn)
            time.sleep(0.1 + random.uniform(0, 0.3))
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


def main():
    parser = argparse.ArgumentParser(description="Zippoom review crawler (sequential mode)")
    parser.add_argument('--headless', action='store_true', help='Run Chrome in headless mode')
    parser.add_argument('--save', type=str, default='리뷰_구조화_결과.csv', help='Output CSV file')
    parser.add_argument('--cookies-file', type=str, default='cookies.json', help='Path to cookies file to save/load')
    parser.add_argument('--record-cookies', action='store_true', help='Open browser for manual login and save cookies to --cookies-file')
    parser.add_argument('--reuse-cookies', action='store_true', help='Load cookies from --cookies-file before crawling')
    parser.add_argument('--profile-dir', type=str, default=TEMP_DATA_PATH, help='Base profile directory to reuse')
    args = parser.parse_args()

    # 쿠키 저장 모드
    if args.record_cookies:
        print("브라우저가 열립니다. 로그인 후 콘솔에서 Enter를 눌러 쿠키를 저장하세요.")
        login_drv = create_driver(headless=False, worker_id=None, base_profile_dir=args.profile_dir)
        try:
            login_drv.get('https://zippoom.com/')
            input('로그인 완료 후 Enter를 누르세요...')
            save_cookies(login_drv, args.cookies_file)
        finally:
            try:
                login_drv.quit()
            except Exception:
                pass
        return

    # 이미 처리된 인덱스 확인
    processed_indices = get_processed_indices(args.save)
    total_rows = len(df)
    
    if processed_indices:
        print(f"이미 처리된 항목: {len(processed_indices)}개")
        print(f"남은 항목: {total_rows - len(processed_indices)}개")
    else:
        print(f"전체 항목: {total_rows}개")

    # 드라이버 생성
    driver = None
    try:
        driver = create_driver(headless=args.headless, worker_id=None, base_profile_dir=args.profile_dir)
        
        # 쿠키 로드
        if args.reuse_cookies and os.path.exists(args.cookies_file):
            print(f"쿠키 로드 중: {args.cookies_file}")
            load_cookies(driver, args.cookies_file, url='https://zippoom.com/')
            if is_logged_in(driver):
                print("✅ 로그인 세션 복원 성공")
            else:
                print("⚠️ 로그인 세션 복원 실패")

        # 순차 처리
        for idx in range(total_rows):
            # 이미 처리된 항목은 건너뛰기
            if idx in processed_indices:
                print(f"[{idx+1}/{total_rows}] 건너뜀 (이미 처리됨)")
                continue

            row = df.iloc[idx]
            kapt_name = row.get('kaptName', '')
            doro_juso = row.get('doroJuso', '')
            
            print(f"\n[{idx+1}/{total_rows}] 크롤링 시작: {kapt_name}")
            
            try:
                # 리뷰 수집
                reviews = crawl_zippoom(doro_juso, kapt_name, driver)
                
                # 각 리뷰에 source_index 추가
                for review in reviews:
                    review['source_index'] = idx
                
                # 리뷰가 없어도 처리 완료 기록
                if not reviews:
                    reviews = [{
                        'kaptName': kapt_name,
                        'doroJuso': doro_juso,
                        'Score': None,
                        'Pros': None,
                        'Cons': None,
                        'source_index': idx
                    }]
                
                # CSV에 즉시 저장
                append_to_csv(reviews, args.save)
                print(f"  ✅ CSV에 저장 완료: {len(reviews)}건")
                
            except Exception as e:
                print(f"  ⚠️ 에러 발생: {e}")
                # 에러가 발생해도 처리 완료로 기록 (무한 루프 방지)
                error_record = [{
                    'kaptName': kapt_name,
                    'doroJuso': doro_juso,
                    'Score': None,
                    'Pros': None,
                    'Cons': None,
                    'source_index': idx,
                    'error': str(e)
                }]
                append_to_csv(error_record, args.save)
            
            # 다음 항목으로 넘어가기 전 대기
            time.sleep(1 + random.uniform(0, 0.3))

    finally:
        if driver:
            try:
                driver.quit()
                print("\n드라이버 종료")
            except Exception:
                pass

    print(f"\n✅ 크롤링 완료! 결과는 '{args.save}'에 저장되었습니다.")


if __name__ == '__main__':
    main()