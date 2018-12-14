from flask import Flask, render_template, url_for, redirect, request
import pandas as pd
import random

app = Flask(__name__)
app.url_map.strict_slashes = False

MEANS = [1710,1711,1712,1713,1714,1715,1716,1717]


@app.route('/label/')
def label():
    # Draw random chamber
    run = random.choice(range(0,99))
    means = MEANS[random.randint(0, len(MEANS)-1)]
    # Check if not scored already
    labels = pd.read_csv('labels-real.csv', names=['run', 'means', 'vfat', 'score'])
    already = labels[(labels.means == means) & (labels.run == run) ]
    if len(already):
        return redirect(url_for('label'))

    # Get the image
    id_name = (str(means) + "_" + str(run)) 
    return render_template('n_label.html', run=run, imgsrc=url_for('static', filename='images/' + id_name + '.png'), id=id_name)

@app.route('/result/')
def result():
    pos = request.args['idplot'].split("_")

    vfats = []
    for i in range(24):
        if 'vf' + str(i+1).zfill(2) in request.args:
            vfats.append((request.args['vf' + str(i+1).zfill(2)] == "on") + 0)
        else:
            vfats.append(0)

    foo = ''
    for x, l in enumerate(vfats):
        csv = pos[1] + "," +pos[0]
        csv += "," + str(x + 1) + "," + str(l)
        foo += "\n" + csv

    with open('labels-real.csv', 'ab') as file_:
        file_.write(foo)

    return redirect(url_for('label'))

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
