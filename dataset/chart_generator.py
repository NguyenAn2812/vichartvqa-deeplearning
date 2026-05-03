import os
import random
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config


def create_chart_and_markdown(cfg, chart_id):
    chart_type = cfg.get('type', 'bar')

    data = {cfg['x_col']: cfg['x_cats']}
    for y_col in cfg['y_cols']:
        try:
            min_v = int(float(y_col.get('min', 10)))
            max_v = int(float(y_col.get('max', 100)))
        except (ValueError, TypeError):
            min_v, max_v = 10, 100
        if min_v >= max_v:
            max_v = min_v + 50
        data[y_col['name']] = [random.randint(min_v, max_v) for _ in cfg['x_cats']]

    df = pd.DataFrame(data)
    md_content = df.to_markdown(index=False)

    fig = plt.figure(figsize=(10, 6), facecolor='white')
    ax = fig.add_subplot(111)
    x_data = df[cfg['x_col']]

    if chart_type == 'pie':
        y_name = cfg['y_cols'][0]['name']
        ax.pie(df[y_name], labels=x_data, autopct='%1.1f%%', startangle=140,
               colors=plt.cm.Pastel1.colors, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
        ax.axis('equal')

    elif chart_type == 'area':
        for y_col in cfg['y_cols']:
            color = y_col.get('color') if y_col.get('color') and '#' in str(y_col.get('color')) else None
            line = ax.fill_between(x_data, df[y_col['name']], alpha=0.3, label=f"Vùng {y_col['name']}", color=color)
            ax.plot(x_data, df[y_col['name']], color=line.get_facecolor()[0], marker='s', markersize=6, linewidth=2.5)

    elif chart_type == 'line':
        for y_col in cfg['y_cols']:
            ax.plot(x_data, df[y_col['name']], label=y_col['name'], marker='o', markersize=8, linewidth=3.5)

    else:
        x_indexes = range(len(x_data))
        width = 0.35
        for i, y_col in enumerate(cfg['y_cols']):
            offset = (i * width) - (width / 2) if len(cfg['y_cols']) > 1 else 0
            ax.bar([x + offset for x in x_indexes], df[y_col['name']], width=width, label=y_col['name'], alpha=0.9)
        ax.set_xticks(x_indexes)
        ax.set_xticklabels(x_data)

    ax.set_title(cfg['title'], fontsize=14, fontweight='bold', pad=25)

    if chart_type != 'pie':
        ax.set_ylabel(cfg['y_label'], fontsize=12)
        ax.legend(loc='upper right', frameon=True, shadow=True)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    img_path = os.path.join(config.IMG_DIR, f"{chart_id}.jpg")
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close(fig)

    md_path = os.path.join(config.MD_DIR, f"{chart_id}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return md_content