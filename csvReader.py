# Reading an excel file using Python
import xlrd
import pandas as pd
import sys

class ExcelFileParser:
    def parseFile(self, inputFilePath, outPutFilePath):
        csv = pd.read_csv(inputFilePath, error_bad_lines=False)

        csv = csv.rename(columns={'Team 1': 'home_team'})
        csv = csv.rename(columns={'Team 2': 'away_team'})

        csv = csv.drop(columns='HT')

        home_score = csv['FT']
        new_home_scores = []
        new_away_scores = []

        for str in home_score:
            ind = str.index('-')
            new_home_scores.append(str[:ind])
            new_away_scores.append(str[ind+1:])

        csv['home_score'] = new_home_scores
        csv['away_score'] = new_away_scores

        csv.to_csv(outPutFilePath, index=None, header=True)

if __name__ == "__main__":
    excelParser = ExcelFileParser()
    excelParser.parseFile("./season/%s" % sys.argv[1], "./season/%s" % sys.argv[2])

