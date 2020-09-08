# GEMDQM

# 1. generate samples

- to generate samples using for training
```
 cd gen
 mkdir Samples
 ./run.sh
 python save_real.py 
 mkdir data
 python root_to_json.py
 mv -r data training
```

# 2. labeling with test data(at local)

- move images to score/static/images
```
 mkdir image
 python pad_to_png.py
```

```
 cd score
 mkdir -p static/images
 python new_label.py
```

- go to `0.0.0.0:5000/label` to start scoring the data. After scoring, scp csv file to server
- merge with this command
```
 ./label-merge.sh
```

# 3. training
```
 cd training
 mkdir figures models-local data // move json and csv file to data directory
 cd codes
 python 4st_fix.py
```
