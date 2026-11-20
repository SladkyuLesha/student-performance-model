"""Render architecture diagrams for the local monitoring system."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from src1.split_data import PROJECT_ROOT

OUT = PROJECT_ROOT / 'reports/figures/architecture'


def box(ax, x, y, w, h, label, fill):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor='#34474e', linewidth=1.3))
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=13, linewidth=1.3, color='#3b5058'))


def save(fig, filename):
    path = OUT / filename
    if path.exists():
        raise FileExistsError(path)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(path)


def main():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set(xlim=(0, 12), ylim=(0, 5)); ax.axis('off')
    box(ax, .2, 3.5, 2.2, .8, 'Пользователь / CSV', '#edf5f2')
    box(ax, .2, 1.2, 2.2, .8, 'Клиент API', '#edf5f2')
    box(ax, 3.1, 3.5, 2.2, .8, 'Панель Streamlit', '#e8f1f5')
    box(ax, 3.1, 1.2, 2.2, .8, 'API FastAPI', '#e8f1f5')
    box(ax, 6.1, 2.3, 2.2, 1, 'Проверка данных\nи прогноз', '#fff3dd')
    box(ax, 9.1, 3.4, 2.4, .9, 'Модели и pipeline', '#f4eaf0')
    box(ax, 9.1, 1.2, 2.4, .9, 'SQLite: история\nи версии', '#eef3e5')
    for start, end in [((2.4,3.9),(3.1,3.9)),((2.4,1.6),(3.1,1.6)),((5.3,3.8),(6.1,3.05)),((5.3,1.6),(6.1,2.55)),((8.3,3.05),(9.1,3.8)),((8.3,2.55),(9.1,1.65))]:
        arrow(ax, start, end)
    ax.set_title('Компоненты локальной системы', fontsize=14)
    save(fig, 'monitoring_components.png')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set(xlim=(0, 12), ylim=(0, 5)); ax.axis('off')
    ax.add_patch(Rectangle((.25,.3),11.4,4.3, facecolor='#fafafa', edgecolor='#34474e', linewidth=1.3))
    ax.text(.5,4.35,'Рабочая станция · локальная сеть 127.0.0.1', fontsize=11)
    box(ax,.65,3.1,2.1,.75,'Браузер', '#edf5f2')
    box(ax,3.5,3.1,2.5,.75,'Streamlit :8501', '#e8f1f5')
    box(ax,.65,1.55,2.1,.75,'Клиент API', '#edf5f2')
    box(ax,3.5,1.55,2.5,.75,'Uvicorn :8000', '#e8f1f5')
    box(ax,7.15,2.3,1.8,.8,'src1/service.py', '#fff3dd')
    box(ax,9.65,3.1,1.55,.75,'joblib', '#f4eaf0')
    box(ax,9.65,1.55,1.55,.75,'SQLite', '#eef3e5')
    for start, end in [((2.75,3.48),(3.5,3.48)),((2.75,1.93),(3.5,1.93)),((6,3.48),(7.15,2.95)),((6,1.93),(7.15,2.45)),((8.95,2.95),(9.65,3.48)),((8.95,2.45),(9.65,1.93))]:
        arrow(ax,start,end)
    ax.set_title('Развёртывание на одной рабочей станции', fontsize=14)
    save(fig, 'monitoring_deployment.png')


if __name__ == '__main__':
    main()
