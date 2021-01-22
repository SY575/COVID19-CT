import os
import shutil

for d in ['train', 'val', 'test']:
    for label in ['no_nCoV', 'nCoV']:
        people_lst = list(set([item.split('_')[0] for item in os.listdir(os.path.join(d, label))]))
        people_lst = {people:f'{d}-{label}-{str(i).zfill(3)}' 
                      for i, people in enumerate(people_lst)}
        for fn in os.listdir(os.path.join(d, label)):
            new_fn = people_lst[fn.split('_')[0]]+'_'+'_'.join(fn.split('_')[1:])
            shutil.move(
                os.path.join(d, label, fn),
                os.path.join(d, label, new_fn))