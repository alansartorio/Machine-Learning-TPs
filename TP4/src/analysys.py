from dataset import load_dataset, budget, genres, imdb_id, original_title, overview, popularity, production_companies, production_countries, release_date, revenue, runtime, spoken_languages, vote_average, vote_count
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Optional

sns.set_theme()

@dataclass
class PlotConfig:
    show: bool = True
    save_path: Optional[str] = None
    tight_layout: bool = False

    def print_plot(self) -> None:
        if self.save_path is not None:
            if self.tight_layout:
                plt.tight_layout()
            plt.savefig(self.save_path)
        if self.show:
            plt.show()


dataset = load_dataset()


def plot_two_variables(x:str,y:str,type:str='scatterplot',config: PlotConfig = PlotConfig()) -> None:
    match type:
        case 'scatterplot':
            sns.scatterplot(data=dataset,x=x,y=y)
        case 'lineplot':
            sns.lineplot(data=dataset,x=x,y=y)
        case 'barplot':
            ax = sns.barplot(data=dataset,x=x,y=y)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    config.print_plot()
    plt.clf()

