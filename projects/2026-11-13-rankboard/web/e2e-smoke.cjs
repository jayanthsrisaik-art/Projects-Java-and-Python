// Two-tab browser smoke test. Needs a running server and Playwright:
//   BASE_URL=http://localhost:8765 node web/e2e-smoke.cjs
const { chromium } = require('playwright');
const BASE = process.env.BASE_URL || 'http://localhost:8765';
(async () => {
  const browser = await chromium.launch();
  const errors = [];
  const pageA = await browser.newPage(); const pageB = await browser.newPage();
  for (const p of [pageA, pageB]) p.on('pageerror', e => errors.push(String(e)));
  const created = await (await pageA.request.post(BASE + '/api/boards', { data: { name: 'E2E board' } })).json();
  const id = created.board.id;
  await pageA.goto(BASE + '/#' + id);
  await pageA.waitForSelector('section.column');
  for (const t of ['alpha', 'beta', 'gamma']) {
    await pageA.fill('section.column:nth-child(1) input', t);
    await pageA.click('section.column:nth-child(1) form button');
    await pageA.waitForFunction(n => document.querySelectorAll('section.column:nth-child(1) li').length === n && !document.querySelector('#status').textContent.includes('pending'), ['alpha','beta','gamma'].indexOf(t)+1);
  }
  await pageB.goto(BASE + '/#' + id);
  await pageB.waitForFunction(() => document.querySelectorAll('li.card').length === 3);
  // A drags gamma to Doing
  await pageA.dragAndDrop('li.card:has-text("gamma")', 'section.column:nth-child(2) ul');
  // A drags alpha below beta
  await pageA.waitForTimeout(400);
  const box = await pageA.locator('li.card:has-text("beta")').boundingBox();
  await pageA.dragAndDrop('li.card:has-text("alpha")', 'li.card:has-text("beta")', { targetPosition: { x: 10, y: box.height - 3 } });
  await pageA.waitForTimeout(500);
  const snap = await (await pageA.request.get(BASE + '/api/boards/' + id)).json();
  const cols = Object.fromEntries(snap.columns.map(c => [c.id, c.name]));
  const serverOrder = snap.cards.map(c => `${cols[c.column_id]}:${c.title}:${c.rank}`);
  console.log('server', serverOrder.join(' | '));
  await pageB.waitForFunction(() => document.querySelector('section.column:nth-child(2) li') !== null, null, { timeout: 5000 });
  await pageB.waitForTimeout(1800);
  const viewB = await pageB.$$eval('section.column', ss => ss.map(s => s.querySelector('h2').textContent + ':' + [...s.querySelectorAll('li .title')].map(x => x.textContent).join(',')));
  console.log('tab B', viewB.join(' | '));
  await pageA.screenshot({ path: process.env.SHOT || 'board.png' });
  console.log('page errors', errors);
  const want = 'To do:beta,alpha | Doing:gamma | Done:';
  if (viewB.join(' | ') !== want || errors.length) { console.error('SMOKE FAILED, expected tab B: ' + want); process.exitCode = 1; }
  else console.log('SMOKE OK');
  await browser.close();
})();
