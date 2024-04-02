import undetected_chromedriver.v2 as uc
import random,time,os,sys
from selenium.webdriver.common.keys import Keys

GMAIL = 'aifinitelearning'
PASSWORD = '8808P9068S'

chrome_options = uc.ChromeOptions()

chrome_options.add_argument("--disable-extensions")

chrome_options.add_argument("--disable-popup-blocking")

chrome_options.add_argument("--profile-directory=Default")

chrome_options.add_argument("--ignore-certificate-errors")

chrome_options.add_argument("--disable-plugins-discovery")

chrome_options.add_argument("--incognito")

chrome_options.add_argument("user_agent=DN")
executable_path= r"C:\Program Files\Google\Chrome\Application\chrome.exe"

driver = uc.Chrome(executable_path=executable_path, options=chrome_options)

driver.delete_all_cookies()

print('done')

time.sleep(10)