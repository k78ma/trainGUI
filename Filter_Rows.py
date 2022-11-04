import csv
with open('grants.csv','r'), open ('filtered_grants.csv','w') as fin, fout:
    writer = csv.writer(fout, delimiter=',')            
    for row in csv.reader(fin, delimiter=','):
        if row[2] == 'Central':
             writer.writerow(row)