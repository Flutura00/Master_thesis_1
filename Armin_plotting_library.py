import numpy as np
import matplotlib.pyplot as mpl
import pylab as pl
import os, sys
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Figure():
    def __init__(self, **opts_dict):

        self.figure_dict = {"figure_title": None,  # Title of plot
                 "plot_title": None, # title of the plot
                 "lc": 'black',  # Standard color of lines
                 "lw": 0.75,  # Standard width of lines
                 "line_dashes": None, # Solid line if none, otherwise provide tupel, for example (2,2)
                 "pt": 'o',  # Standard marker for scatter plots
                 "pc": 'black', # Standard marker color
                 "ps": 2,  # Standard marker size for scatter plots,
                 "fc": 'darkgray', # Fillcolor of polygons
                 "elw":0.5,  # Standard edge line width of scatter plots and errorbars
                 "ec": 'darkgray',  # Standard edge color of makers in scatter plots
                 "alpha": 1,  # Transparency
                 "errorbar_area": True,  # Shaded error bar in line plots
                 "textcolor":'black',  # Color of all texts and labnels
                 "fig_width": 21.00,  # Width of the entire figure in cm (21.59 for US-letter)
                 "fig_height": 29.70,  # Hight of the entire figure in cm (27.94 for US-letter)
                 "helper_lines_dashes": (2, 2),  # Length of dashes for horizontal and vertical helper lines
                 "helper_lines_lw": 0.25, # Linewidth of the helperlines, default is half of the axes_lw
                 "helper_lines_lc": "darkgray",  # Color of horizontal and vertical helper lines, and polar grids
                 "xl_distance": 0.75,  # Distance (in cm) of the xlabel from the axes
                 "yl_distance": 0.75,  # Distance (in cm) of the ylabel from the axes
                 "fontsize": 6,  # Font size of all text labels
                 "fontsize_figure_title": 12,  # Font size of the figure title on the top of the page
                 "fontsize_plot_label": 8,  # Font size of the bold plot labels
                 "dpi": 600,
                 "rasterized": False, # Display plots as pixel rasters (for huge data clouds, this makes the plot easier to render and load)
                 "axes_lw": 0.5, # Linewidth of all plot borders, helper lines will be half of that
                 "facecolor": 'none', # The background color of the whole plot
                 "fontname": 'Arial',
                 "fontfamily": 'sans-serif',
                 "xmin": None,
                 "xmax": None,
                 "ymin": None,
                 "ymax": None,
                 "zmin": None,
                 "zmax": None,
                 "label": None,
                 "xticks": None,
                 "yticks": None,
                 "zticks": None,
                 "xl": None,
                 "yl": None,
                 "zl": None,
                 "vertical_bar_width": 0.8,
                 "horizontal_bar_height": 0.8,
                 "xticklabels": None,
                 "xticklabels_rotation": 0,
                 "yticklabels": None,
                 "yticklabels_rotation": 0,
                 "zticklabels": None,
                 "zticklabels_rotation": 0,
                 "vspans": None,
                 "hspans": None,
                 "hlines": None,
                 "helper_lines_alpha": 1.0,
                 "vlines": None,
                 "xlog": False,
                 "ylog": False,
                 "axis_off": False,
                 "legend_xpos": None,
                 "legend_ypos": None,
                 "colormap": "inferno",
                 "norm_colormap": None,
                 "show_colormap": None,
                 "image_interpolation": "bilinear",
                 "image_origin": "lower"}

        for key in opts_dict.keys():
            self.figure_dict[key] = opts_dict[key]

        mpl.rcParams['font.sans-serif'] = self.figure_dict["fontname"]
        mpl.rcParams['font.family'] = self.figure_dict["fontfamily"]
        mpl.rcParams['font.size'] = self.figure_dict["fontsize"]
        mpl.rcParams['figure.dpi'] = self.figure_dict["dpi"]
        mpl.rcParams['pdf.fonttype'] = 42  # Always embedd fonts in the pdf
        mpl.rcParams['ps.fonttype'] = 42

        # Figure size must be in inches
        self.fig = pl.figure(num=None, figsize=[self.figure_dict["fig_width"] / 2.54, self.figure_dict["fig_height"] / 2.54],
                             facecolor=self.figure_dict["facecolor"], edgecolor='none', dpi=self.figure_dict["dpi"])

        if self.figure_dict["figure_title"] is not None:
            pl.figtext(0.5, 0.95, self.figure_dict["figure_title"], ha='center', va='center', fontsize=self.figure_dict["fontsize_figure_title"], color=self.figure_dict["textcolor"])

    def create_plot(self, **opts_dict):
        return Plot(self, opts_dict)

    def create_polar_plot(self, **opts_dict):
        return PolarPlot(self, opts_dict)

    def show(self):
        self.fig.show()

    def save(self, path, tight=False, open_file=False):
        if tight:
            self.fig.savefig(path, bbox_inches='tight', pad_inches=0.1, facecolor=self.figure_dict["facecolor"], edgecolor='none',
                             transparent=True)
        else:
            self.fig.savefig(path, facecolor=self.figure_dict["facecolor"], edgecolor='none', transparent=True)

        if open_file:
            if sys.platform.startswith('darwin'):
                os.system(f"open '{path}'")
            else:
                os.startfile(path)

        pl.close()

    def add_text(self, x, y, text, rotation=0, ha='center'):
        pl.figtext(x / self.figure_dict["fig_width"], y / self.figure_dict["fig_height"], text, ha=ha, ma=ha, va='center',
                   color=self.figure_dict["textcolor"], rotation=rotation)


class SuperPlot():
    def __init__(self, figure, opts_dict):

        self.figure = figure
        self.plot_dict = self.figure.figure_dict.copy()

        for key in opts_dict.keys():
            self.plot_dict[key] = opts_dict[key]

        if self.plot_dict["legend_xpos"] is None:
            self.plot_dict["legend_xpos"] = self.plot_dict["xpos"] + self.plot_dict["plot_width"]
        if self.plot_dict["legend_ypos"] is None:
            self.plot_dict["legend_ypos"] = self.plot_dict["ypos"] + self.plot_dict["plot_height"]

        self.current_zorder = 0

    def set_axes_properties(self):
        if self.plot_dict["xmin"] is not None:
            self.ax.set_xlim([self.plot_dict["xmin"], self.plot_dict["xmax"]])
        if self.plot_dict["ymin"] is not None:
            self.ax.set_ylim([self.plot_dict["ymin"], self.plot_dict["ymax"]])

        if self.plot_dict["xl"] is not None:
            self.ax.set_xlabel(self.plot_dict["xl"], horizontalalignment='center', verticalalignment='center',
                               color=self.plot_dict["textcolor"])
            x_coord = (self.plot_dict["xpos"] + 0.5 * self.plot_dict["plot_width"]) / self.plot_dict["fig_width"]
            y_coord = (self.plot_dict["ypos"] - self.plot_dict["xl_distance"]) / self.plot_dict["fig_height"]

            self.ax.xaxis.set_label_coords(x_coord, y_coord, self.figure.fig.transFigure)

        if self.plot_dict["yl"] is not None:
            self.ax.set_ylabel(self.plot_dict["yl"], verticalalignment='center', horizontalalignment='center',
                               color=self.plot_dict["textcolor"])

            x_coord = (self.plot_dict["xpos"] - self.plot_dict["yl_distance"]) / self.plot_dict["fig_width"]
            y_coord = (self.plot_dict["ypos"] + 0.5 * self.plot_dict["plot_height"]) / self.plot_dict["fig_height"]

            self.ax.yaxis.set_label_coords(x_coord, y_coord, self.figure.fig.transFigure)

        if self.plot_dict["xlog"] is True:
            self.ax.set_xscale("log")

        if self.plot_dict["ylog"] is True:
            self.ax.set_yscale("log")

        if self.plot_dict["axis_off"] is True:
            self.ax.set_axis_off()

    def draw_horizontal_significance_label(self, x0, x1, y, label):
        self.current_zorder += 1
        self.ax.plot([x0, x1], [y, y], color=self.plot_dict["textcolor"], lw=self.plot_dict["axes_lw"], alpha=1.0, solid_capstyle="round",
                     rasterized=self.plot_dict["rasterized"], zorder=self.current_zorder)

        self.ax.text((x0 + x1)/2, y, label, ha='center', ma='center', va='bottom', color=self.plot_dict["textcolor"], rotation=0, zorder=self.current_zorder)

    def draw_vertical_significance_label(self, y0, y1, x, label):
        self.current_zorder += 1
        self.ax.plot([x, x], [y0, y1], color=self.plot_dict["textcolor"], lw=self.plot_dict["axes_lw"], alpha=1.0, solid_capstyle="round",
                     rasterized=self.plot_dict["rasterized"], zorder=self.current_zorder)

        self.ax.text(x, (y0 + y1)/2, " "+label, ha='left', ma='left', va='center', color=self.plot_dict["textcolor"], rotation=0., zorder=self.current_zorder)

    def draw_text(self, x, y, text, rotation=0, ha='center'):
        self.ax.text(x, y, text, ha=ha, ma=ha, va='center', color=self.plot_dict["textcolor"], rotation=rotation)

    def draw_line(self, x, y, **opts_dict):
        artist_dict = self.plot_dict.copy()

        for key in opts_dict.keys():
            artist_dict[key] = opts_dict[key]

        if "yerr" in artist_dict:
            yerr = np.array(artist_dict["yerr"])
            self.current_zorder += 1
            if artist_dict["errorbar_area"] is False:
                self.ax.errorbar(x, y, yerr=yerr, elinewidth=artist_dict["elw"], ecolor=artist_dict["lc"], fmt='none', capsize=1.5*artist_dict["elw"], mew=artist_dict["elw"],
                                 solid_capstyle='round', solid_joinstyle='round', rasterized=artist_dict["rasterized"], zorder=self.current_zorder)
            else:
                self.ax.fill_between(x, y - yerr, y + yerr, lw=0, edgecolor='none', facecolor=artist_dict["lc"], alpha=0.2,
                                     rasterized=artist_dict["rasterized"], zorder=self.current_zorder)

        if "xerr" in artist_dict:
            xerr = np.array(artist_dict["xerr"])
            self.current_zorder += 1
            if artist_dict["errorbar_area"] is False:
                self.ax.errorbar(x, y, xerr=xerr, elinewidth=artist_dict["elw"], ecolor=artist_dict["lc"], fmt='none', capsize=1.5*artist_dict["elw"], mew=artist_dict["elw"],
                                 solid_capstyle='round', solid_joinstyle='round', rasterized=artist_dict["rasterized"], zorder=self.current_zorder)
            else:
                self.ax.fill_betweenx(y, x + xerr, x - xerr, lw=0, edgecolor='none', facecolor=artist_dict["lc"], alpha=0.2,
                                      rasterized=artist_dict["rasterized"], zorder=self.current_zorder)

        self.current_zorder += 1

        if artist_dict["line_dashes"] is not None:
            self.ax.plot(x, y, color=artist_dict["lc"], lw=artist_dict["lw"], alpha=artist_dict["alpha"], dashes=artist_dict["line_dashes"], dash_capstyle="round",
                         label=artist_dict["label"], rasterized=artist_dict["rasterized"], zorder=self.current_zorder)
        else:
            self.ax.plot(x, y, color=artist_dict["lc"], lw=artist_dict["lw"], alpha=artist_dict["alpha"], solid_capstyle="round", label=artist_dict["label"],
                         rasterized=artist_dict["rasterized"], zorder=self.current_zorder)

        if artist_dict["label"] is not None:
            leg = self.ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(
                artist_dict["legend_xpos"] / artist_dict["fig_width"],
                artist_dict["legend_ypos"] / artist_dict["fig_height"]),
                                    bbox_transform=self.figure.fig.transFigure)

            for text in leg.get_texts():
                pl.setp(text, color=artist_dict["textcolor"])

    def draw_polygon(self, x, y, **opts_dict):
        artist_dict = self.plot_dict.copy()

        for key in opts_dict.keys():
            artist_dict[key] = opts_dict[key]

        self.current_zorder += 1
        # TODO If you want this to work maybe dont initalize it with None
        if type(artist_dict["line_dashes"]) != type(None) :
            self.ax.fill(x, y, fc=artist_dict["fc"], alpha=artist_dict["alpha"], ec=None, lw=None, label=artist_dict["label"], rasterized=artist_dict["rasterized"], zorder=self.current_zorder)
            self.ax.plot(x, y, color=artist_dict["lc"], lw=artist_dict["lw"], dashes=artist_dict["line_dashes"], dash_capstyle="round", rasterized=artist_dict["rasterized"], zorder=self.current_zorder)

        else:
            self.ax.fill(x, y, fc=artist_dict["fc"], alpha=artist_dict["alpha"], ec=None, lw=None, label=artist_dict["label"], rasterized=artist_dict["rasterized"], zorder=self.current_zorder)
            self.ax.plot(x, y, color=artist_dict["lc"], lw=artist_dict["lw"], solid_capstyle="round", rasterized=artist_dict["rasterized"], zorder=self.current_zorder)

        if artist_dict["label"] is not None:
            leg = self.ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(
                artist_dict["legend_xpos"] / artist_dict["fig_width"], artist_dict["legend_ypos"] / artist_dict["fig_height"]),
                                    bbox_transform=self.fig.fig.transFigure)

            for text in leg.get_texts():
                pl.setp(text, color=artist_dict["textcolor"])

    def draw_scatter(self, x, y, **opts_dict):
        artist_dict = self.plot_dict.copy()

        for key in opts_dict.keys():
            artist_dict[key] = opts_dict[key]

        if "yerr" in artist_dict:
            self.current_zorder += 1
            self.ax.errorbar(x, y, yerr=artist_dict["yerr"], elinewidth=artist_dict["elw"], ecolor=artist_dict["ec"], fmt='none',
                             capsize=1.5*artist_dict["elw"], mew=artist_dict["elw"],
                             solid_capstyle='round', solid_joinstyle='round', zorder=self.current_zorder)

        if "xerr" in artist_dict:
            self.current_zorder += 1
            self.ax.errorbar(x, y, xerr=artist_dict["xerr"], elinewidth=artist_dict["elw"], ecolor=artist_dict["ec"], fmt='none',
                             capsize=1.5*artist_dict["elw"], mew=artist_dict["elw"],
                             solid_capstyle='round', solid_joinstyle='round', zorder=self.current_zorder)

        # to get the same units as seaborn, we need to use the square of the marker size
        self.current_zorder += 1

        if isinstance(artist_dict["pc"], list) or isinstance(artist_dict["pc"], np.ndarray):
            if artist_dict["norm_colormap"] is None:
                self.ax.scatter(x, y, c=artist_dict["pc"], marker=artist_dict["pt"], s=artist_dict["ps"] ** 2, linewidths=artist_dict["elw"], edgecolor=artist_dict["ec"], alpha=artist_dict["alpha"], label=artist_dict["label"],
                                rasterized=artist_dict["rasterized"], cmap=artist_dict["colormap"], norm=None, vmin=artist_dict["zmin"], vmax=artist_dict["zmax"], zorder=self.current_zorder)
            else:

                self.ax.scatter(x, y, c=artist_dict["pc"], marker=artist_dict["pt"], s=artist_dict["ps"] ** 2,
                                linewidths=artist_dict["elw"], edgecolor=artist_dict["ec"], alpha=artist_dict["alpha"],
                                label=artist_dict["label"],
                                rasterized=artist_dict["rasterized"], cmap=artist_dict["colormap"], norm=artist_dict["norm_colormap"],
                                zorder=self.current_zorder)

        else:
            self.ax.scatter(x, y, c=artist_dict["pc"], marker=artist_dict["pt"], s=artist_dict["ps"] ** 2,
                            linewidths=artist_dict["elw"], edgecolor=artist_dict["ec"],
                            alpha=artist_dict["alpha"], label=artist_dict["label"],
                            rasterized=artist_dict["rasterized"], zorder=self.current_zorder)

        if artist_dict["label"] is not None:
            leg = self.ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(
                artist_dict["legend_xpos"] / artist_dict["fig_width"], artist_dict["legend_ypos"] / artist_dict["fig_height"]),
                                    bbox_transform=self.figure.fig.transFigure)

            for text in leg.get_texts():
                pl.setp(text, color=artist_dict["textcolor"])

    def draw_vertical_bars(self, x, y, **opts_dict):
        artist_dict = self.plot_dict.copy()

        for key in opts_dict.keys():
            artist_dict[key] = opts_dict[key]

        if type(artist_dict["lw"]) is not list:
            artist_dict["lw"] = [artist_dict["lw"]] * len(x)

        if type(artist_dict["lc"]) is not list:
            artist_dict["lc"] = [artist_dict["lc"]] * len(x)

        if type(artist_dict["alpha"]) is not list:
            artist_dict["alpha"] = [artist_dict["alpha"]] * len(x)

        for i in range(len(x)):

            if "yerr" in artist_dict:
                self.current_zorder += 1
                self.ax.errorbar(x[i], y[i], yerr=artist_dict["yerr"][i], elinewidth=artist_dict["elw"], ecolor=artist_dict["lc"][i], fmt='none',
                                 rasterized=artist_dict["rasterized"],
                                 capsize=artist_dict["elw"] * 1.5, mew=artist_dict["elw"], solid_capstyle='round', solid_joinstyle='round',
                                 zorder=self.current_zorder)

            self.current_zorder += 1

            self.ax.bar(x[i], y[i], edgecolor=None, lw=None, alpha=artist_dict["alpha"][i], facecolor=artist_dict["lc"][i], rasterized=artist_dict["rasterized"],
                        align='center', width=artist_dict["vertical_bar_width"], label=artist_dict["label"] if i == 0 and artist_dict["label"] is not None else None, zorder=self.current_zorder)

            if "bl" in artist_dict:
                if not np.isnan(y[0]) and not np.isnan(x[0]):
                    x_ = self.ax.xpos / artist_dict["fig_width"] + artist_dict["plot_width"] * (
                            (x[0] - self.ax.xmin) / (self.ax.xmax - self.ax.xmin)) / artist_dict["fig_width"]
                    y_ = self.ax.ypos / artist_dict["fig_height"] + artist_dict["plot_heighth"] * (
                            (y[0] + np.sign(y[0] + artist_dict["yerr"][0]) * artist_dict["yerr"][0] - self.ax.ymin) / (
                        self.ax.ymax - self.ax.ymin)) / artist_dict["fig_height"] + np.sign(
                        y[0] + artist_dict["yerr"][0]) * 0.2 / artist_dict["fig_height"]

                    pl.figtext(x_, y_, artist_dict["bl"], ha='center', va='center', color=artist_dict["textcolor"])

        if artist_dict["label"] is not None:
            leg = self.ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(
                artist_dict["legend_xpos"] / artist_dict["fig_width"], artist_dict["legend_ypos"] / artist_dict["fig_height"]),
                                    bbox_transform=self.figure.fig.transFigure)

            for text in leg.get_texts():
                pl.setp(text, color=artist_dict["textcolor"])

    def draw_horizontal_bars(self, x, y, **opts_dict):

        artist_dict = self.plot_dict.copy()

        for key in opts_dict.keys():
            artist_dict[key] = opts_dict[key]

        if type(artist_dict["lw"]) is not list:
            artist_dict["lw"] = [artist_dict["lw"]] * len(x)

        if type(artist_dict["lc"]) is not list:
            artist_dict["lc"] = [artist_dict["lc"]] * len(x)

        if type(artist_dict["alpha"]) is not list:
            artist_dict["alpha"] = [artist_dict["alpha"]] * len(x)

        for i in range(len(x)):
            self.current_zorder += 1
            self.ax.barh(y[i], x[i], edgecolor=None, lw=None, facecolor=artist_dict["lc"][i], alpha=artist_dict["alpha"][i], align='center',
                         rasterized=True,
                         height=artist_dict["horizontal_bar_height"], label=artist_dict["label"] if i == 0 and artist_dict["label"] is not None else None, zorder=self.current_zorder)

            if "xerr" in artist_dict:
                self.current_zorder += 1
                self.ax.errorbar(x[i], y[i], xerr=artist_dict["xerr"][i], elinewidth=artist_dict["elw"], ecolor=artist_dict["lc"][i], fmt='none',
                                 capsize=artist_dict["elw"] * 1.5, rasterized=artist_dict["rasterized"],
                                 mew=artist_dict["elw"], solid_capstyle='round', solid_joinstyle='round', zorder=self.current_zorder)

    def draw_swarmplot(self, ys, **opts_dict):

        artist_dict = self.plot_dict.copy()

        for key in opts_dict.keys():
            artist_dict[key] = opts_dict[key]

        if type(artist_dict["pc"]) is not list:
            artist_dict["pc"] = [artist_dict["pc"]] * len(artist_dict["xticklabels"])

        lc = sns.color_palette(artist_dict["pc"])

        ys = [pd.Series(y) for y in ys]

        df = pd.concat(ys, axis=1, keys=range(len(artist_dict["xticks"]))).stack(0)
        df = df.reset_index(level=1)
        df.columns = ["X", "Y"]

        self.current_zorder += 1
        g = sns.swarmplot(x="X", y="Y", data=df, ax=self.ax, palette=lc, hue="X", edgecolor=artist_dict["ec"],
                      linewidth=artist_dict["elw"], rasterized=artist_dict["rasterized"],
                      marker=artist_dict["pt"], size=artist_dict["ps"], alpha=artist_dict["alpha"], zorder=self.current_zorder)
        g.get_legend().remove()

        # As seanborn replaces the labels, set them again
        self.ax.set_xticklabels(self.plot_dict["xticklabels"])
        self.ax.set_yticklabels(self.plot_dict["yticklabels"])
        self.ax.set_xlabel(self.plot_dict["xl"])
        self.ax.set_ylabel(self.plot_dict["yl"])

        self.set_axes_properties()

    def draw_violinplot(self, ys, **opts_dict):

        artist_dict = self.plot_dict.copy()

        for key in opts_dict.keys():
            artist_dict[key] = opts_dict[key]

        if type(artist_dict["alpha"]) is not list:
            artist_dict["alpha"] = [artist_dict["alpha"]] * len(artist_dict["xticks"])

        if type(artist_dict["pc"]) is not list:
            artist_dict["pc"] = [artist_dict["pc"]] * len(artist_dict["xticks"])

        palette = sns.color_palette(artist_dict["pc"])

        ys_ = [pd.Series(y) for y in ys]

        df = pd.concat(ys_, axis=1, keys=range(len(artist_dict["xticks"]))).stack(0)
        df = df.reset_index(level=1)
        df.columns = ["X", "Y"]

        self.current_zorder += 1
        g = sns.violinplot(x="X", y="Y", data=df, ax=self.ax, palette=palette, hue=None, edgecolor=artist_dict["ec"],
                       linewidth=artist_dict["elw"], inner=None, rasterized=artist_dict["rasterized"], zorder=self.current_zorder)
        #g.get_legend().remove()
        pl.setp(self.ax.collections[-len(ys):], alpha=artist_dict["alpha"])

        # As seanborn replaces the labels, set them again
        self.ax.set_xticklabels(self.plot_dict["xticklabels"])
        self.ax.set_yticklabels(self.plot_dict["yticklabels"])
        self.ax.set_xlabel(self.plot_dict["xl"])
        self.ax.set_ylabel(self.plot_dict["yl"])

        self.set_axes_properties()

    def draw_boxplot(self, ys, **opts_dict):

        artist_dict = self.plot_dict.copy()

        for key in opts_dict.keys():
            artist_dict[key] = opts_dict[key]

        if type(artist_dict["pc"]) is not list:
            artist_dict["pc"] = [artist_dict["pc"]] * len(artist_dict["xticks"])

        palette = sns.color_palette(artist_dict["pc"])

        ys_ = [pd.Series(y) for y in ys]

        df = pd.concat(ys_, axis=1, keys=range(len(artist_dict["xticks"]))).stack(0)
        df = df.reset_index(level=1)
        df.columns = ["X", "Y"]

        self.current_zorder += 1
        flierprops = dict(marker=artist_dict["pt"],
                          markersize=artist_dict["ps"],
                          zorder=self.current_zorder,
                          markerfacecolor=artist_dict["textcolor"],
                          linestyle='none',
                          markeredgecolor=artist_dict["textcolor"])

        g = sns.boxplot(x="X", y="Y", data=df, ax=self.ax, palette=palette, hue=None,
                    boxprops={"zorder": self.current_zorder, "alpha": artist_dict["alpha"]},
                    linewidth=artist_dict["elw"],
                    width=artist_dict["vertical_bar_width"],
                    flierprops=flierprops,
                    zorder=self.current_zorder)
        #g.get_legend().remove()

        # As seaborn replaces the labels, set them again
        self.ax.set_xticklabels(self.plot_dict["xticklabels"])
        self.ax.set_yticklabels(self.plot_dict["yticklabels"])
        self.ax.set_xlabel(self.plot_dict["xl"])
        self.ax.set_ylabel(self.plot_dict["yl"])

        self.set_axes_properties()

    def draw_image(self, img, extent, **opts_dict):
        artist_dict = self.plot_dict.copy()

        for key in opts_dict.keys():
            artist_dict[key] = opts_dict[key]

        self.current_zorder += 1
        if artist_dict["norm_colormap"] is None:
            self.ax.imshow(np.array(img), extent=extent, interpolation=artist_dict["image_interpolation"], origin=artist_dict["image_origin"], aspect='auto',
                           cmap=pl.get_cmap(artist_dict["colormap"]), vmin=artist_dict["zmin"], vmax=artist_dict["zmax"], alpha=artist_dict["alpha"],
                           rasterized=artist_dict["rasterized"], zorder=self.current_zorder)
        else:
            self.ax.imshow(np.array(img), extent=extent, interpolation=artist_dict["image_interpolation"],
                           origin=artist_dict["image_origin"], aspect='auto',
                           cmap=pl.get_cmap(artist_dict["colormap"]), norm=artist_dict["norm_colormap"], alpha=artist_dict["alpha"],
                           rasterized=artist_dict["rasterized"], zorder=self.current_zorder)

    def draw_pcolormesh(self, x, y, Z, aa=False, shading="gouraud", **opts_dict):
        artist_dict = self.plot_dict.copy()

        for key in opts_dict.keys():
            artist_dict[key] = opts_dict[key]

        self.current_zorder += 1
        if artist_dict["norm_colormap"] is None:
            self.ax.pcolormesh(x, y, Z, aa=aa, shading=shading,
                               cmap=pl.get_cmap(artist_dict["colormap"]),
                               rasterized=artist_dict["rasterized"],
                               vmin=artist_dict["zmin"],
                               vmax=artist_dict["zmax"],
                               alpha=artist_dict["alpha"],
                               zorder=self.current_zorder)
        else:
            self.ax.pcolormesh(x, y, Z, aa=aa, shading=shading,
                               cmap=pl.get_cmap(artist_dict["colormap"]),
                               rasterized=artist_dict["rasterized"],
                               norm=artist_dict["norm_colormap"],
                               alpha=artist_dict["alpha"],
                               zorder=self.current_zorder)

        # if type(whitebar) == list: #whitebar[0] = length of one stimuli; whitebar[1] = length of hole; whitebar[2] = no of stimuli #whitebar=[120,10,6]
        #     layer = np.full_like(mesh, False)
        #     for i in range(whitebar[2]-1):
        #         temp_start = (whitebar[0]*i + whitebar[0]) + (whitebar[1]*i)
        #         temp_end = temp_start + 10
        #         layer[:,temp_start:temp_end] = 1
        #     masked = np.ma.masked_where(layer == 0, layer)
        #     self.ax.pcolormesh(masked, alpha=1, cmap="gray_r")
        #


class PolarPlot(SuperPlot):
    def __init__(self, figure, opts_dict):
        SuperPlot.__init__(self, figure, opts_dict)

        self.current_zorder = 0

        self.ax = self.figure.fig.add_axes([self.plot_dict["xpos"] / self.plot_dict["fig_width"],
                                         self.plot_dict["ypos"] / self.plot_dict["fig_height"],
                                         self.plot_dict["plot_width"] / self.plot_dict["fig_width"],
                                         self.plot_dict["plot_height"] / self.plot_dict["fig_height"]], polar=True)

        self.ax.set_facecolor("none")
        self.ax.spines['polar'].set_linewidth(self.plot_dict["axes_lw"])  # Line width of plots
        self.ax.spines['polar'].set_color(self.plot_dict["textcolor"])
        self.ax.spines['polar'].set_zorder(100) # Make sure lines are always above all plot elements

        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)

        self.current_zorder += 1
        self.ax.grid(color=self.plot_dict["helper_lines_lc"], linewidth=self.plot_dict["axes_lw"]*0.5, dashes=self.plot_dict["helper_lines_dashes"],
                     solid_capstyle="round", dash_capstyle="round", zorder=self.current_zorder)

        self.ax.set_rticks(self.plot_dict["yticks"], zorder=100)
        self.ax.set_rlim([self.plot_dict["ymin"], self.plot_dict["ymax"]])

        if self.plot_dict["xticklabels"] is None:
            self.plot_dict["xticklabels"] = [str(lbl) for lbl in self.plot_dict["xticks"]]

        if self.plot_dict["yticklabels"] is None:
            self.plot_dict["yticklabels"] = [str(lbl) for lbl in self.plot_dict["yticks"]]

        for i in range(len(self.plot_dict["xticklabels"])):
            self.plot_dict["xticklabels"][i] = self.plot_dict["xticklabels"][i].replace("-", u'–')

        for i in range(len(self.plot_dict["yticklabels"])):
            self.plot_dict["yticklabels"][i] = self.plot_dict["yticklabels"][i].replace("-", u'–')

        self.ax.set_xlim([self.plot_dict["xmin"], self.plot_dict["xmax"]])
        self.ax.set_xticks(self.plot_dict["xticks"], zorder=100)

        if self.plot_dict["xticklabels_rotation"] == 0:
            self.ax.set_xticklabels(self.plot_dict["xticklabels"], rotation=self.plot_dict["xticklabels_rotation"], horizontalalignment='center',
                                    color=self.plot_dict["textcolor"])
        else:
            self.ax.set_xticklabels(self.plot_dict["xticklabels"], rotation=self.plot_dict["xticklabels_rotation"], horizontalalignment='right',
                                    color=self.plot_dict["textcolor"])

        if self.plot_dict["yticklabels_rotation"] == 0:
            self.ax.set_yticklabels(self.plot_dict["yticklabels"], rotation=self.plot_dict["yticklabels_rotation"], horizontalalignment='center',
                                    color=self.plot_dict["textcolor"])
        else:
            self.ax.set_yticklabels(self.plot_dict["yticklabels"], rotation=self.plot_dict["yticklabels_rotation"], horizontalalignment='right',
                                    color=self.plot_dict["textcolor"])

        self.current_zorder += 1
        tick = [self.ax.get_rmax(), self.ax.get_rmax() * 0.97]
        for t in self.plot_dict["xticks"]:
            self.ax.plot([t, t], tick, lw=self.plot_dict["axes_lw"], color=self.plot_dict["textcolor"], zorder=self.current_zorder)

        pl.figtext((self.plot_dict["xpos"] - 0.3 * self.plot_dict["fontsize"] / 9.) / self.plot_dict["fig_width"],
                   (self.plot_dict["ypos"] + self.plot_dict["plot_height"] + 0.5) / self.plot_dict["fig_height"], self.plot_dict["plot_label"], weight='bold',
                   fontsize=self.plot_dict["fontsize_plot_label"], ha='center', va='center', color=self.plot_dict["textcolor"])

        self.set_axes_properties()

class Plot(SuperPlot):
    def __init__(self, figure, opts_dict):
        SuperPlot.__init__(self, figure, opts_dict)

        self.ax = self.figure.fig.add_axes([self.plot_dict["xpos"] / self.plot_dict["fig_width"],
                                         self.plot_dict["ypos"] / self.plot_dict["fig_height"],
                                         self.plot_dict["plot_width"] / self.plot_dict["fig_width"],
                                         self.plot_dict["plot_height"] / self.plot_dict["fig_height"]])

        self.ax.set_facecolor("none")

        self.ax.spines['right'].set_color('none')
        self.ax.spines['top'].set_color('none')

        self.ax.spines['left'].set_linewidth(self.plot_dict["axes_lw"])  # Line width of plots
        self.ax.spines['bottom'].set_linewidth(self.plot_dict["axes_lw"])
        self.ax.spines['left'].set_color(self.plot_dict["textcolor"])
        self.ax.spines['bottom'].set_color(self.plot_dict["textcolor"])
        self.ax.spines["left"].set_zorder(100)
        self.ax.spines["bottom"].set_zorder(100)

        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.yaxis.set_ticks_position('left')
        self.ax.tick_params('both', width=self.plot_dict["axes_lw"], which='major', tickdir="out", color=self.plot_dict["textcolor"], zorder=100)
        self.ax.tick_params('both', width=self.plot_dict["axes_lw"], which='minor', tickdir="out", color=self.plot_dict["textcolor"], zorder=100)

        if self.plot_dict["xticks"] is not None:
            self.ax.set_xticks(self.plot_dict["xticks"])

            if self.plot_dict["xticklabels"] is None:
                self.plot_dict["xticklabels"] = [str(lbl) for lbl in self.plot_dict["xticks"]]

        if self.plot_dict["xticklabels"] is not None:
            self.plot_dict["xticklabels"] = self.plot_dict["xticklabels"].copy()  # make sure we do not modifiy the original

            for i in range(len(self.plot_dict["xticklabels"])):
                self.plot_dict["xticklabels"][i] = self.plot_dict["xticklabels"][i].replace("-", '–')

            if self.plot_dict["xticklabels_rotation"] == 0:
                self.ax.set_xticklabels(self.plot_dict["xticklabels"], rotation=self.plot_dict["xticklabels_rotation"],
                                        horizontalalignment='center',
                                        color=self.plot_dict["textcolor"])
            else:
                self.ax.set_xticklabels(self.plot_dict["xticklabels"], rotation=self.plot_dict["xticklabels_rotation"],
                                        horizontalalignment='right',
                                        color=self.plot_dict["textcolor"])

        else:
            self.ax.spines['bottom'].set_visible(False)
            self.ax.tick_params(axis='x', which='minor', bottom='off')
            self.ax.tick_params(axis='x', which='major', bottom='off')
            self.ax.get_xaxis().set_ticks([])

        if self.plot_dict["yticks"] is not None:
            self.ax.set_yticks(self.plot_dict["yticks"])

            if self.plot_dict["yticklabels"] is None:
                self.plot_dict["yticklabels"] = [str(lbl) for lbl in self.plot_dict["yticks"]]

        if self.plot_dict["yticklabels"] is not None:
            self.plot_dict["yticklabels"] = self.plot_dict["yticklabels"].copy()  # make sure we do not modifiy the original

            for i in range(len(self.plot_dict["yticklabels"])):
                self.plot_dict["yticklabels"][i] = self.plot_dict["yticklabels"][i].replace("-", '–')

            self.ax.set_yticklabels(self.plot_dict["yticklabels"], rotation=self.plot_dict["yticklabels_rotation"], horizontalalignment='right',
                                    color=self.plot_dict["textcolor"])
        else:
            self.ax.spines['left'].set_visible(False)
            self.ax.tick_params(axis='y', which='minor', left='off')
            self.ax.tick_params(axis='y', which='major', left='off')
            self.ax.get_yaxis().set_ticks([])

        if self.plot_dict["vspans"] is not None:
            self.current_zorder += 1
            for vspan in self.plot_dict["vspans"]:
                self.ax.axvspan(vspan[0], vspan[1], lw=0, edgecolor='none', facecolor=vspan[2],
                                alpha=vspan[3], zorder=self.current_zorder)

        if self.plot_dict["hspans"] is not None:
            self.current_zorder += 1
            for hspan in self.plot_dict["hspans"]:
                self.ax.axhspan(hspan[0], hspan[1], lw=0, edgecolor='none', facecolor=hspan[2],
                                alpha=hspan[3], zorder=self.current_zorder)

        if self.plot_dict["hlines"] is not None:
            self.current_zorder += 1
            for hline in self.plot_dict["hlines"]:
                self.ax.axhline(hline, linewidth=self.plot_dict["helper_lines_lw"], color=self.plot_dict["helper_lines_lc"],
                                dashes=self.plot_dict["helper_lines_dashes"], alpha=self.plot_dict["helper_lines_alpha"],
                                solid_capstyle="round", dash_capstyle="round", zorder=self.current_zorder)

        if self.plot_dict["vlines"] is not None:
            self.current_zorder += 1
            for vline in self.plot_dict["vlines"]:
                self.ax.axvline(vline, linewidth=self.plot_dict["helper_lines_lw"], color=self.plot_dict["helper_lines_lc"],
                                dashes=self.plot_dict["helper_lines_dashes"], alpha=self.plot_dict["helper_lines_alpha"],
                                solid_capstyle="round", dash_capstyle="round", zorder=self.current_zorder)

        if self.plot_dict["plot_title"] is not None:
            self.ax.set_title(self.plot_dict["plot_title"], color=self.plot_dict["textcolor"], fontsize=self.plot_dict["fontsize"])

        if self.plot_dict["plot_label"] is not None:
            if self.ax.spines['left'].get_visible():
                pl.figtext((self.plot_dict["xpos"] - 1.8 * self.plot_dict["fontsize"] / 9.) / self.plot_dict["fig_width"],
                           (self.plot_dict["ypos"] + self.plot_dict["plot_height"] + 0.5) / self.plot_dict["fig_height"], self.plot_dict["plot_label"], weight='bold',
                           fontsize=self.plot_dict["fontsize_plot_label"], ha='center', va='center', color=self.plot_dict["textcolor"])
            else:
                pl.figtext((self.plot_dict["xpos"] - 0.3 * self.plot_dict["fontsize"] / 9.) /  self.plot_dict["fig_width"],
                           (self.plot_dict["ypos"] + self.plot_dict["plot_height"] + 0.5) / self.plot_dict["fig_height"], self.plot_dict["plot_label"], weight='bold',
                           fontsize=self.plot_dict["fontsize_plot_label"], ha='center', va='center', color=self.plot_dict["textcolor"])

        # Draw the colormap next to it
        if self.plot_dict["show_colormap"]:
            cbar_ax = self.figure.fig.add_axes([(self.plot_dict["xpos"] + self.plot_dict["plot_width"] + self.plot_dict["plot_width"] / 20.) / self.plot_dict["fig_width"],
                                                 self.plot_dict["ypos"] / self.plot_dict["fig_height"],
                                                 (self.plot_dict["plot_width"] / 10.) / self.plot_dict["fig_width"],
                                                 self.plot_dict["plot_height"] / self.plot_dict["fig_height"]], frameon=False)

            cbar_ax.yaxis.set_ticks([])
            cbar_ax.xaxis.set_ticks([])

            cbar_ax2 = cbar_ax.twinx()
            cbar_ax2.set_facecolor("none")

            for val in ["left", "right", "bottom", "top"]:
                 cbar_ax2.spines[val].set_zorder(100) # Always above all plot elements
                 cbar_ax2.spines[val].set_linewidth(self.plot_dict["axes_lw"])
                 cbar_ax2.spines[val].set_color(self.plot_dict["textcolor"])

            cbar_ax2.tick_params('both', width=self.plot_dict["axes_lw"], which='major', tickdir="out", color=self.plot_dict["textcolor"], zorder=100)
            cbar_ax2.tick_params('both', width=self.plot_dict["axes_lw"], which='minor', tickdir="out", color=self.plot_dict["textcolor"], zorder=100)

            self.current_zorder += 1
            if self.plot_dict["norm_colormap"] is None:
                cbar_ax2.imshow(np.c_[np.linspace(self.plot_dict["zmin"], self.plot_dict["zmax"], 500)],
                                extent=(0, 1, self.plot_dict["zmin"], self.plot_dict["zmax"]), alpha=self.plot_dict["alpha"],
                                rasterized=self.plot_dict["rasterized"],
                                aspect='auto', origin='lower', cmap=pl.get_cmap(self.plot_dict["colormap"]), zorder=self.current_zorder)
            else:
                cbar_ax2.imshow(np.c_[np.linspace(self.plot_dict["norm_colormap"].vmin, self.plot_dict["norm_colormap"].vmax, 500)],
                                extent=(0, 1, self.plot_dict["norm_colormap"].vmin, self.plot_dict["norm_colormap"].vmax),
                                alpha=self.plot_dict["alpha"],
                                rasterized=self.plot_dict["rasterized"],
                                aspect='auto', origin='lower', norm=self.plot_dict["norm_colormap"],
                                cmap=pl.get_cmap(self.plot_dict["colormap"]), zorder=self.current_zorder)

            if self.plot_dict["zticks"] is not None:
                 cbar_ax2.set_yticks(self.plot_dict["zticks"])

                 if self.plot_dict["zticklabels"] is None:
                     self.plot_dict["zticklabels"] = [str(lbl) for lbl in self.plot_dict["zticks"]]

                 for i in range(len(self.plot_dict["zticklabels"])):
                     self.plot_dict["zticklabels"][i] = self.plot_dict["zticklabels"][i].replace("-", '–')

                 cbar_ax2.set_yticklabels(self.plot_dict["zticklabels"], rotation=self.plot_dict["zticklabels_rotation"],
                                          horizontalalignment='left',
                                          color=self.plot_dict["textcolor"])

            if self.plot_dict["zl"] is not None:
                 cbar_ax2.set_ylabel(self.plot_dict["zl"], color=self.plot_dict["textcolor"])

        self.set_axes_properties()