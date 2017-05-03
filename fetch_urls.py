import os
import sys
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

def fetch_urls(channel):
  driver = webdriver.Chrome()
  url = os.path.join('https://www.youtube.com/user', channel, 'videos')
  driver.get(url)

  while True:
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')

    try:
      more = WebDriverWait(driver, 3).until(
        EC.element_to_be_clickable((By.CLASS_NAME, 'load-more-button'))
      )
    except StaleElementReferenceException:
      continue;
    except TimeoutException:
      break;

    more.click()

  selector = '.yt-thumb-default .yt-thumb-clip img'
  elements = driver.find_elements_by_css_selector(selector)
  src_list = [element.get_attribute('src') for element in elements]
  driver.quit()

  with open(f'urls/{channel}.txt', 'wt') as f:
    for src in src_list:
      print(src, file=f)

if __name__ == '__main__':
  fetch_urls(sys.argv[1])
