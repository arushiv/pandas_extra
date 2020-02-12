import pandas
import numpy
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import glob
import time
from scipy import stats
import pybedtools
from openpyxl import load_workbook
from adjustText import adjust_text

class ExtraFunctions(object):
    def __init__(self, figure_directory):
        self.fdir = figure_directory

    def scatter_adjust_text(self, df, x, y, label, arrowprops={}):
        props = {'arrowstyle':'->',
                 'color':'black'}
        props.update(arrowprops)
        
        scatter_adjust_texts = []
        for i, point in df.iterrows():
            scatter_adjust_texts.append(plt.text(point[x],
                                  point[y],
                                  str(point[label]), size=8))
            
        adjust_text(scatter_adjust_texts, arrowprops=props)
            
    def get_cdf_column(self, d, col, label=""):
        """from a dataframe and column name with counts data, plot ECDF
        Get values, use numpy.histogram and numpy.cumsum. Starts with 0"""

        minx = 0
        maxx = max(d[col])

        n_s  = d[col].values

        hist = numpy.histogram(n_s, bins=len(range(0, maxx)))

        o = numpy.cumsum(hist[0]/n_s.size)

        X = numpy.array(range(0,maxx))

        g = plt.plot(X, o, label=label)
        return g

    def get_numeric_cumsum(self, d, col, label=""):
        """from a dataframe and column name with counts data, plot cumulative sum.
        Get values, use numpy.histogram and numpy.cumsum. Starts with 0. Option to add
        label to the plot for a legend"""

        minx = 0
        maxx = max(d[col])

        n_s  = d[col].values

        hist = numpy.histogram(n_s, bins=len(range(0, maxx)))

        o = numpy.cumsum(hist[0])

        X = numpy.array(range(0,maxx))

        g = plt.plot(X, o, label=label)
        return g


    def get_rows_greater_than(self, d, col, start=1000, step=100, end=10000, label=""):
        """from a dataframe and column name with counts data, plot number of rows where column col has a value >= x
        Default start  = 1000, step=100, end=10000 """
        p = pandas.DataFrame(numpy.arange(start, end, step), columns=['x'])
        p['y'] = p['x'].map(lambda x: len(d[d[col]>=x]))
        g = plt.plot(p['x'], p['y'], label=label)
        return g
        
    def legend_out(self):
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


    def step_plot(self, d, col):
        """For counts column of a dataframe d, make a cumulative step plot"""
        minx = 0
        maxx = max(d[col])
        n_s  = d.sort_values('dna')['dna'].values
        s = numpy.arange(n_s.size)

        g = plt.step(n_s, s)
        return g


    def gp(self, name):
        return os.path.join(self.fdir, name)

    def rot(self, g):
        for i, ax in enumerate(g.fig.axes):   ## getting all axes of the fig object
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)

    def save(self, name):
        """Save figure fdir directory  """
        plt.savefig(os.path.join(self.fdir, name), bbox_inches="tight")

    def saveb(self, name):
        """Save both pdf and png figures. Gets the base name of the provided name by
        removing the trailing .pdf or .png. fdir (figure directory) assigned when calss initiated """
        if name.endswith(".pdf") or name.endswith(".png"):
            name = name[:-4]
            plt.savefig(os.path.join(self.fdir, f"{name}.pdf"), bbox_inches="tight")
            plt.savefig(os.path.join(self.fdir, f"{name}.png"), bbox_inches="tight")


    def symmetric_scatter(self, g):
        """Make x and y axes of a scatter plot equal ranges,
        also plot an x=y line"""
        M = round(max(g.axes.get_xlim()[1], g.axes.get_ylim()[1]) + 0.5, 0)
        plt.xlim(0, M)
        plt.ylim(0, M)
        plt.plot([0, M], [0, M], linestyle='--', color='black')


    def multi_excel(self, filename):
        """Use pandas.to_excel to have multiple sheets on the same workbook
        Provide filename; use this function to append sheets to existing workbook.
        The function returns a writer object which can be used in the pandas.to_excel
        function as the argument, along with a sheet_name argument
        If the filename does not exits, it makes one by saving an empty dataframe to
        the filename. So the workbook will have a leading empty sheet named Sheet1
        Usage: writer = pe.multi_excel(my.xlsx)
        pandas.DataFrame().to_excel(writer, sheet_name="new")"""

        if not os.path.exists(filename):
            pandas.DataFrame().to_excel(filename)
        book = load_workbook(filename)
        writer = pandas.ExcelWriter(filename, engine='openpyxl')
        writer.book = book
        return writer


    def explode_df(self, d, lst_col, delimiter):
        """For a column with delimited entries, split by delimiter and copy rest of the columns
        into new rows"""
        # convert column values to list
        x = d.assign(**{lst_col:full[lst_col].str.split(delimiter)})
        df = pandas.DataFrame({col:numpy.repeat(x[col].values, x[lst_col].str.len())
                               for col in x.columns.difference([lst_col])
        }).assign(**{lst_col:numpy.concatenate(x[lst_col].values)})[x.columns.tolist()]
        return df


    def copy_df_by_list(self, d, vlist, colname):
        dflist = []
        
        for l in vlist:
            d1 = d.copy()
            d1[colname] = l
            dflist.append(d1)
        
        out = pandas.concat(dflist)
        return out
                        
