const text = `Here is the data:
| 项目 (Item) | 重量 (Mass) | 热量 (Calories) | 脂肪 (Fat) | 碳水 (Carbs) | 蛋白质 (Protein) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 汉堡 (burger) | 382.47 g | 530.83 kcal | 10.39 g | 44.99 g | 28.75 g |
| 薯条 (french fries) | 108.44 g | 242.77 kcal | 2.35 g | 16.18 g | 6.61 g |
| 总计 (Total) | 490.91 g | 773.6 kcal | 12.74 g | 61.17 g | 35.36 g |
Enjoy!`;

const tableRegex = /((?:\|.*\|\n)+)/g;
const matches = text.match(tableRegex);
console.log(matches);
