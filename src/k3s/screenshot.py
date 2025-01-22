from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Set Chrome options
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--headless")  # Run without GUI (optional)
chrome_options.add_argument("--user-data-dir=/tmp/selenium_profile")  # Unique directory

# Start WebDriver with options
driver = webdriver.Chrome(options=chrome_options)

# Open Spekt8 UI
driver.get("http://localhost:3000")

# Save screenshot
driver.save_screenshot("spekt8_diagram.png")

# Close browser
driver.quit()

print("Screenshot saved as spekt8_diagram.png")

