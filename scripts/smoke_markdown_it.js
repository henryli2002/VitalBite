/* eslint-disable */
// Headless smoke: verify renderMarkdown output by shimming a minimal `window`
// and loading the vendored markdown-it + the frontend helpers we care about.
// Run: node scripts/smoke_markdown_it.js

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const FRONT = path.join(__dirname, '..', 'src', 'frontend');
const mdSrc = fs.readFileSync(path.join(FRONT, 'vendor', 'markdown-it.min.js'), 'utf8');

// Fake DOM bits used by escapeHtml and state look-ups.
const fakeDoc = {
    createElement: () => {
        const el = { innerHTML: '' };
        Object.defineProperty(el, 'textContent', {
            set(v) {
                // Minimal text -> HTML escape, matches browser behaviour for our cases.
                el.innerHTML = String(v)
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;');
            },
            get() { return ''; },
        });
        return el;
    },
};

const ctx = {
    window: {},
    document: fakeDoc,
    state: { activeUserId: 'usr_smoke' },
    console,
};
ctx.self = ctx.window;
ctx.globalThis = ctx;
vm.createContext(ctx);

// Load markdown-it into the shimmed window.
vm.runInContext(mdSrc, ctx, { filename: 'markdown-it.min.js' });
// The UMD bundle attaches to globalThis when available, which in a vm context
// is the context object itself. Mirror it onto the fake window so the
// app.js code path (which does `window.markdownit`) finds it.
if (typeof ctx.markdownit === 'function' && typeof ctx.window.markdownit !== 'function') {
    ctx.window.markdownit = ctx.markdownit;
}
if (typeof ctx.window.markdownit !== 'function') {
    console.error('FAIL: markdown-it did not attach to window');
    process.exit(1);
}

// Extract just the helpers we need from app.js (avoid executing the full file,
// which depends on a browser DOM). We eval the needed function sources directly.
const appSrc = fs.readFileSync(path.join(FRONT, 'app.js'), 'utf8');

function extract(name, source) {
    const re = new RegExp(
        `function ${name}\\s*\\([^)]*\\)\\s*\\{[\\s\\S]*?^\\}`,
        'm',
    );
    const m = source.match(re);
    if (!m) throw new Error(`Could not extract ${name}`);
    return m[0];
}

const helpers = [
    'escapeHtml',
    'buildRestaurantCards',
    'buildNutritionViz',
    '_getMd',
    'renderMarkdown',
    'parseNutritionVal',
    'inferFoodType',
    'getTotalFromRow',
    'computeGoalDeltaNote',
].map((n) => {
    try { return extract(n, appSrc); } catch (_) { return null; }
}).filter(Boolean).join('\n');

// Some of the nutrition helpers reference globals we haven't stubbed; we only
// need them not to throw on basic text cases. Guard by try/catch.
vm.runInContext(`
    var _chartIdCounter = 0;
    var config = { activeUserId: 'usr_smoke' };
    ${helpers}
    // Stub nutrition viz — full renderer pulls in chart state we don't care to shim here.
    function buildNutritionViz(items) {
        const rows = Array.isArray(items) ? items.length : 0;
        return '<div class="nutrition-viz" data-rows="' + rows + '"></div>';
    }
    var _md = null;
`, ctx, { filename: 'app-helpers-eval' });

function check(label, input, expectParts) {
    const out = vm.runInContext(`renderMarkdown(${JSON.stringify(input)})`, ctx);
    const missing = expectParts.filter((p) => !out.includes(p));
    if (missing.length) {
        console.error(`[FAIL] ${label}\n  missing: ${missing.join(' | ')}\n  output: ${out}`);
        process.exitCode = 1;
    } else {
        console.log(`[ok]   ${label}`);
    }
}

// 1. Plain text with bold + list
check('plain markdown', 'hello **world**\n- one\n- two',
    ['<strong>world</strong>', '<li>one</li>', '<li>two</li>', '<ul>']);

// 2. Generic table uses .md-table classes
check('generic table', '| A | B |\n|---|---|\n| 1 | 2 |',
    ['<div class="table-wrapper"><table class="md-table">', '<th>A</th>', '<td>1</td>']);

// 3. ```nutrition fenced block rendered as the nutrition viz widget
const nutritionBlock = '识别结果：\n```nutrition\n[{"name":"汉堡","weight_g":200,"calories":500,"fat_g":20,"carbs_g":40,"protein_g":25}]\n```';
check('nutrition fence', nutritionBlock,
    ['nutrition-viz', 'data-rows="1"']);

// 3b. Malformed nutrition fence falls back to default code block (no widget)
check('malformed nutrition fence', '```nutrition\nnot json\n```',
    ['<code']);

// 3c. A plain Markdown nutrition-style table is NOT swapped anymore — it stays a table.
check('plain nutrition table is no longer swapped', '| 食物 | 重量 | 热量 | 脂肪 | 碳水 | 蛋白质 |\n|---|---|---|---|---|---|\n| 汉堡 | 200g | 500kcal | 20g | 40g | 25g |',
    ['<table class="md-table">', '<th>食物</th>']);

// 4. ```restaurants fenced block rendered as cards
const restBlock = '附近推荐：\n```restaurants\n[{"name":"Sushi One","address":"1 Main St","rating":4.3,"cuisine":"日料","health":"healthy","advice":"清汤拉面为佳"}]\n```';
check('restaurants fence', restBlock,
    ['restaurant-cards', 'rc-card', 'Sushi One', 'rc-health-good']);

// 5. Malformed restaurants fence falls back to default code block
check('malformed restaurants fence', '```restaurants\nnot json\n```',
    ['<code']);  // default fenced renderer

// 6. User prose with fake `[image: xxxx]` does NOT trigger anything special
check('no injection via [image:', 'hello [image: aaaa] ok',
    ['hello [image: aaaa] ok']);

if (process.exitCode) {
    console.error('\nSOME CHECKS FAILED');
} else {
    console.log('\nAll checks passed');
}
