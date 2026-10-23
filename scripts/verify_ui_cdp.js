const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const brainDir = 'C:\\Users\\Satishreddy Padala\\.gemini\\antigravity\\brain\\1fecde03-3864-45db-9479-2c623fb9e11f';
const userDataDir = path.join(process.env.TEMP, 'chrome_cdp_anti_flicker_' + Math.random().toString(36).substring(2));

async function main() {
  const chromeProc = spawn(chromePath, [
    '--headless=new',
    '--disable-gpu',
    '--remote-debugging-port=9223',
    '--window-size=1600,2400',
    `--user-data-dir=${userDataDir}`,
    'https://nseuniverse.duckdns.org/?module=options'
  ]);

  await new Promise(r => setTimeout(r, 2500));

  const res = await fetch('http://127.0.0.1:9223/json/list');
  const targets = await res.json();
  const pageTarget = targets.find(t => t.type === 'page') || targets[0];
  const wsUrl = pageTarget.webSocketDebuggerUrl;

  const ws = new WebSocket(wsUrl);
  let id = 1;
  const pending = new Map();
  const exceptions = [];
  const logs = [];

  function send(method, params = {}) {
    return new Promise((resolve) => {
      const msgId = id++;
      pending.set(msgId, resolve);
      ws.send(JSON.stringify({ id: msgId, method, params }));
    });
  }

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.method === 'Runtime.exceptionThrown') {
      exceptions.push(msg.params.exceptionDetails);
    }
    if (msg.method === 'Runtime.consoleAPICalled') {
      const text = msg.params.args.map(a => a.value || a.description || '').join(' ');
      logs.push(text);
    }
    if (msg.id && pending.has(msg.id)) {
      const resolve = pending.get(msg.id);
      pending.delete(msg.id);
      resolve(msg.result);
    }
  };

  await new Promise((resolve) => (ws.onopen = resolve));
  await send('Page.enable');
  await send('Runtime.enable');

  console.log('Connected to Chrome CDP. Waiting 5s for live options ticks to populate...');
  await new Promise(r => setTimeout(r, 5000));

  // Screenshot 1: Options Desk
  const shot1 = await send('Page.captureScreenshot', { format: 'png' });
  const shot1Path = path.join(brainDir, 'live_options_desk_flicker_free.png');
  fs.writeFileSync(shot1Path, Buffer.from(shot1.data, 'base64'));
  console.log(`[OK] Captured Options Desk: ${shot1Path} (${fs.statSync(shot1Path).size} bytes)`);

  // Navigate to Intraday Desk
  console.log('Navigating to Intraday Desk...');
  await send('Page.navigate', { url: 'https://nseuniverse.duckdns.org/?module=intraday' });
  await new Promise(r => setTimeout(r, 5000));

  // Screenshot 2: Intraday Desk
  const shot2 = await send('Page.captureScreenshot', { format: 'png' });
  const shot2Path = path.join(brainDir, 'live_intraday_desk_flicker_free.png');
  fs.writeFileSync(shot2Path, Buffer.from(shot2.data, 'base64'));
  console.log(`[OK] Captured Intraday Desk: ${shot2Path} (${fs.statSync(shot2Path).size} bytes)`);

  ws.close();
  chromeProc.kill();
  try {
    fs.rmSync(userDataDir, { recursive: true, force: true });
  } catch (e) {}

  console.log('\n--- VERIFICATION REPORT ---');
  console.log(`Page Exceptions: ${exceptions.length}`);
  if (exceptions.length > 0) {
    exceptions.forEach(e => console.log('EXCEPTION:', e.text, e.exception?.description));
  } else {
    console.log('[PASS] ZERO page exceptions detected during live streaming!');
  }
}

main().catch(err => {
  console.error('CDP script error:', err);
  process.exit(1);
});
