const match = `| 项目 (Item) | 重量 (Mass) | 热量 (Calories) | 脂肪 (Fat) | 碳水 (Carbs) | 蛋白质 (Protein) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 汉堡 (burger) | 382.47 g | 530.83 kcal | 10.39 g | 44.99 g | 28.75 g |
| 薯条 (french fries) | 108.44 g | 242.77 kcal | 2.35 g | 16.18 g | 6.61 g |
| **总计 (Total)** | **490.91 g** | **773.6 kcal** | **12.74 g** | **61.17 g** | **35.36 g** |`;

const lines = match.trim().split('\n');
const header = lines[0];
const isNutrition = header.includes('热量') || header.includes('Calories') || header.includes('重量');

if (isNutrition) {
    let cardsHtml = '<div class="nutrition-cards-container">';
    for (let i = 2; i < lines.length; i++) {
        const cols = lines[i].split('|').slice(1, -1).map(c => c.trim());
        if (cols.length >= 6) {
            const isTotal = cols[0].includes('总计') || cols[0].includes('Total');
            const cardClass = isTotal ? 'nutrition-card total-card' : 'nutrition-card';
            
            const clean = str => str.replace(/\*\*/g, '').trim();
            cardsHtml += `
            <div class="${cardClass}">
                <div class="nc-header">${clean(cols[0])}</div>
                <div class="nc-stats">
                    <span class="nc-btn nc-cal">🔥 ${clean(cols[2])}</span>
                    <span class="nc-btn nc-mass">⚖️ ${clean(cols[1])}</span>
                    <span class="nc-btn nc-macros">🥑 脂 ${clean(cols[3])} • 🍞 碳 ${clean(cols[4])} • 🥩 蛋 ${clean(cols[5])}</span>
                </div>
            </div>`;
        }
    }
    cardsHtml += '</div>';
    console.log(cardsHtml);
}
