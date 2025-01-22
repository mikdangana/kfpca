from selenium import webdriver

# Start WebDriver (Ensure Chrome is installed)
driver = webdriver.Chrome()

# Open Spekt8 UI
driver.get("http://localhost:3000")

# Save screenshot
driver.save_screenshot("spekt8_diagram.png")

# Close browser
driver.quit()

print("Screenshot saved as spekt8_diagram.png")

