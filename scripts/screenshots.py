"""
Automated Screenshot Capture Script for DeliveryETA
Uses Playwright to capture verified, high-resolution screenshots of the Streamlit application:
- 01-order-input-form.png: Initial user input form with features and sliders
- 02-prediction-result.png: Real-time estimated delivery time prediction
- 03-model-performance.png: Overfitting verification dashboard (Train vs Test MAE, RMSE, R2)
- 04-full-dashboard.png: Full-page layout of the application
"""

import argparse
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright


def wait_for_server(url, max_attempts=30, delay=2):
    import urllib.request
    print(f"Waiting for application to be available at {url}...")
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=3) as response:
                if response.status == 200:
                    print(f"Server is healthy (attempt {attempt})!")
                    return True
        except Exception:
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    if response.status == 200:
                        print(f"Server is reachable (attempt {attempt})!")
                        return True
            except Exception:
                pass
        time.sleep(delay)
    raise RuntimeError(f"Server at {url} did not become available after {max_attempts * delay} seconds.")


def capture_screenshots(app_url, output_dir):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {app_url}...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
        context = browser.new_context(
            viewport={'width': 1280, 'height': 900},
            device_scale_factor=2
        )
        page = context.new_page()

        print(f"Navigating to {app_url}...")
        page.goto(app_url, wait_until='networkidle')
        page.wait_for_selector('h2:has-text("Enter Order Details"), h1:has-text("Enter Order Details")', timeout=30000)
        time.sleep(2)  # Allow Streamlit animations to settle

        # 1. Capture initial form
        form_shot = out_path / "01-order-input-form.png"
        page.screenshot(path=str(form_shot))
        print(f"Saved {form_shot}")

        # 2. Trigger prediction
        print("Interacting with prediction button...")
        predict_btn = page.locator('button:has-text("Predict Delivery Time")')
        predict_btn.scroll_into_view_if_needed()
        time.sleep(1)
        predict_btn.click()

        # Wait for the prediction banner to appear
        page.wait_for_selector('div[data-testid="stAlert"]', timeout=15000)
        time.sleep(1.5)

        prediction_shot = out_path / "02-prediction-result.png"
        page.screenshot(path=str(prediction_shot))
        print(f"Saved {prediction_shot}")

        # 3. Model Performance metrics section
        perf_header = page.locator('h2:has-text("Model Performance"), h1:has-text("Model Performance")')
        perf_header.scroll_into_view_if_needed()
        time.sleep(1.5)

        perf_shot = out_path / "03-model-performance.png"
        page.screenshot(path=str(perf_shot))
        print(f"Saved {perf_shot}")

        # 4. Full page screenshot
        full_shot = out_path / "04-full-dashboard.png"
        page.screenshot(path=str(full_shot), full_page=True)
        print(f"Saved {full_shot}")

        browser.close()
    print("Screenshot capture completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Capture DeliveryETA Streamlit app screenshots")
    parser.add_argument("--url", default="http://localhost:8501", help="Base URL of the Streamlit application")
    parser.add_argument("--output-dir", default="docs/screenshots", help="Directory where screenshots are saved")
    args = parser.parse_args()

    wait_for_server(args.url)
    capture_screenshots(args.url, args.output_dir)
