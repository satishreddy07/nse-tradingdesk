const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('web/templates/index.html', 'utf8');
const regex = /<script\b[^>]*>([\s\S]*?)<\/script>/gi;
let match;
let count = 0;
let errors = 0;

while ((match = regex.exec(html)) !== null) {
  count++;
  const js = match[1];
  // Skip external scripts with src or json templates
  if (!js.trim()) continue;
  try {
    new vm.Script(js);
  } catch (err) {
    errors++;
    console.error(`❌ Syntax error in script tag #${count}:`, err.message);
  }
}

if (errors === 0) {
  console.log(`✅ All ${count} script tags in web/templates/index.html validated successfully with 0 errors!`);
  process.exit(0);
} else {
  console.error(`Found ${errors} syntax errors.`);
  process.exit(1);
}
