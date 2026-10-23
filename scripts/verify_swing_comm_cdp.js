const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const brainDir = 'C:\\Users\\Satishreddy Padala\\.gemini\\antigravity\\brain\\1fecde03-3864-45db-9479-2c623fb9e11f';
const userDataDir = path.join(process.env.TEMP, 'chrome_cdp_extra_' + Math.random().toString(36).substring(2));

async function main() {
  const chromeProc = spawn(chromePath, [
    '--headless=new',
    '--disable-gpu',
    '--remote-debugging-port=9224',
    '--window-size=1600,2200',
    `--user-data-dir=${userDataDir}`,
    'https://nseuniverse.duckdns.org/?module=swing'
  ]);

  await new Promise(r => setTimeout(r, 2500));
  const res = await fetch('http://127.0.0.1:9224/json/list');
  const targets = await res.json();
  const pageTarget = targets.find(t => t.type === 'page') || targets[0];
  const ws = new WebSocket(pageTarget.webSocketDebuggerUrl);

  let id = 1;
  const pending = new Map();
  function send(method, params = {}) {
    return new Promise(resolve => {
      const msgId = id++;
      pending.set(msgId, resolve);
      ws.send(JSON.stringify({ id: msgId, method, params }));
    });
  }

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      const resolve = pending.get(msg.id);
      pending.delete(msg.id);
      resolve(msg.result);
    }
  };

  await new Promise(r => (ws.onopen = r));
  await send('Page.enable');
  await new Promise(r => setTimeout(r, 4000));

  // Screenshot: Swing Desk
  const shotSwing = await send('Page.captureScreenshot', { format: 'png' });
  const p1 = path.join(brainDir, 'live_swing_desk_flicker_free.png');
  fs.writeFileSync(p1, Buffer.from(shotSwing.data, 'base64'));
  console.log('[OK] Captured Swing Desk:', p1);

  // Navigate to Commodities Desk
  await send('Page.navigate', { url: 'https://nseuniverse.duckdns.org/?module=commodities' });
  await new Promise(r => setTimeout(r, 4000));

  // Screenshot: Commodities Desk
  const shotComm = await send('Page.captureScreenshot', { format: 'png' });
  const p2 = path.join(brainDir, 'live_commodities_desk_flicker_free.png');
  fs.writeFileSync(p2, Buffer.from(shotComm.data, 'base64'));
  console.log('[OK] Captured Commodities Desk:', p2);

  ws.close();
  chromeProc.kill();
  try { fs.rmSync(userDataDir, { recursive: true, force: true }); } catch (e) {}
}

main().catch(console.error);
