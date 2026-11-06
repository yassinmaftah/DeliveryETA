/**
 * Automated Screenshot Capture Script using Chrome DevTools Protocol (CDP)
 * Runs directly inside container with zero npm dependencies.
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const TARGET_URL = process.env.APP_URL || 'http://deliveryeta_app:8501';
const OUTPUT_DIR = path.resolve(__dirname, '../docs/screenshots');
const CHROME_PATH = process.env.CHROME_PATH || '/ms-playwright/chromium-1140/chrome-linux/chrome';

if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForHttp(url, maxRetries = 30) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch (e) {}
    await sleep(500);
  }
  throw new Error(`Failed to connect to ${url}`);
}

class CDPClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.id = 1;
    this.callbacks = new Map();
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.wsUrl);
      this.ws.onopen = () => resolve();
      this.ws.onerror = (err) => reject(err);
      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.id && this.callbacks.has(msg.id)) {
          const { resolve, reject } = this.callbacks.get(msg.id);
          this.callbacks.delete(msg.id);
          if (msg.error) {
            reject(new Error(msg.error.message));
          } else {
            resolve(msg.result);
          }
        }
      };
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.id++;
      this.callbacks.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async eval(expression) {
    const res = await this.send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    return res.result ? res.result.value : null;
  }

  async captureScreenshot(outputPath, fullPage = false) {
    const params = { format: 'png' };
    if (fullPage) {
      params.captureBeyondViewport = true;
    }
    const res = await this.send('Page.captureScreenshot', params);
    const buffer = Buffer.from(res.data, 'base64');
    fs.writeFileSync(outputPath, buffer);
    console.log(`Saved screenshot: ${outputPath} (${(buffer.length / 1024).toFixed(1)} KB)`);
  }

  close() {
    if (this.ws) this.ws.close();
  }
}

async function main() {
  console.log('====================================================');
  console.log('Automated Screenshot Capture for DeliveryETA');
  console.log(`Target URL: ${TARGET_URL}`);
  console.log(`Output Dir: ${OUTPUT_DIR}`);
  console.log('====================================================');

  console.log('Starting headless Chromium...');
  const chrome = spawn(CHROME_PATH, [
    '--headless=new',
    '--remote-debugging-port=9222',
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--disable-gpu',
    '--window-size=1280,900',
  ]);

  chrome.on('error', (err) => console.error('Chrome process error:', err));

  try {
    await waitForHttp('http://127.0.0.1:9222/json/version');
    console.log('Chromium debugging port ready.');

    // Create a new tab target with TARGET_URL
    const newTargetRes = await fetch(`http://127.0.0.1:9222/json/new?${encodeURIComponent(TARGET_URL)}`, { method: 'PUT' });
    const targetInfo = await newTargetRes.json();
    console.log(`Created page target. WebSocket URL: ${targetInfo.webSocketDebuggerUrl}`);

    const client = new CDPClient(targetInfo.webSocketDebuggerUrl);
    await client.connect();
    console.log('Connected to CDP WebSocket.');

    await client.send('Page.enable');
    await client.send('Runtime.enable');
    await client.send('DOM.enable');

    // Wait for Streamlit app to connect and render
    console.log('Waiting for Streamlit app to render...');
    let ready = false;
    for (let i = 0; i < 40; i++) {
      await sleep(1000);
      const text = await client.eval(`document.body ? document.body.innerText : ''`);
      if (text && text.includes('Enter Order Details') && text.includes('Model Performance')) {
        ready = true;
        console.log(`Streamlit app rendered successfully after ${i + 1}s.`);
        break;
      }
    }

    if (!ready) {
      console.warn('Warning: Streamlit elements not fully confirmed, continuing anyway...');
    }

    await sleep(2000); // Allow styles and charts to fully paint

    // Screenshot 1: Order input form
    console.log('1. Capturing Order Input Form...');
    await client.eval(`window.scrollTo(0, 0);`);
    await sleep(500);
    await client.captureScreenshot(path.join(OUTPUT_DIR, '01-order-input-form.png'));

    // Step 2: Trigger prediction
    console.log('2. Clicking "Predict Delivery Time" button...');
    const clicked = await client.eval(`(() => {
      const buttons = Array.from(document.querySelectorAll('button'));
      const btn = buttons.find(b => b.innerText.includes('Predict Delivery Time'));
      if (btn) {
        btn.click();
        return true;
      }
      return false;
    })()`);
    console.log(`Button clicked: ${clicked}`);

    // Wait for prediction alert to appear
    console.log('Waiting for prediction result...');
    for (let i = 0; i < 15; i++) {
      await sleep(1000);
      const alertText = await client.eval(`(() => {
        const el = document.querySelector('[data-testid="stAlert"]');
        return el ? el.innerText : '';
      })()`);
      if (alertText && alertText.includes('Estimated Delivery Time')) {
        console.log(`Prediction received: "${alertText}"`);
        break;
      }
    }
    await sleep(1500);

    // Screenshot 2: Prediction result
    console.log('Capturing Prediction Result...');
    await client.eval(`(() => {
      const el = document.querySelector('[data-testid="stAlert"]') || document.querySelector('button');
      if (el) el.scrollIntoView({ behavior: 'instant', block: 'center' });
    })()`);
    await sleep(1000);
    await client.captureScreenshot(path.join(OUTPUT_DIR, '02-prediction-result.png'));

    // Step 3: Model Performance section
    console.log('3. Scrolling to Model Performance metrics...');
    await client.eval(`(() => {
      const headers = Array.from(document.querySelectorAll('h1, h2, h3'));
      const header = headers.find(h => h.innerText.includes('Model Performance'));
      if (header) header.scrollIntoView({ behavior: 'instant', block: 'start' });
      else window.scrollTo(0, document.body.scrollHeight);
    })()`);
    await sleep(1500);
    await client.captureScreenshot(path.join(OUTPUT_DIR, '03-model-performance.png'));

    // Step 4: Full page layout
    console.log('4. Capturing Full Dashboard...');
    await client.eval(`window.scrollTo(0, 0);`);
    await sleep(1000);
    await client.captureScreenshot(path.join(OUTPUT_DIR, '04-full-dashboard.png'), true);

    client.close();
    console.log('All screenshots captured successfully!');
  } finally {
    chrome.kill('SIGTERM');
  }
}

main().catch((err) => {
  console.error('Fatal screenshot script error:', err);
  process.exit(1);
});
