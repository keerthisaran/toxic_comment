
import sys
print(sys.executable)
import preprocess_text_ksc as pp
import pandas as pd
import config
from sklearn import model_selection

pipeline1=[  
            str.lower,
            pp.remove_accented_chars,
            pp.get_expanded,
            # pp.remove_html_tags,
            pp.remove_urls,
            pp.repl_white_newline,
            pp.sep_punct_for_tokens
          ]


def preprocess(df,
               folds,
               label,
               x_col,
               processed_x_col,
               shuffle,
               fold_col='kfold',
               save_file_path=None):
      df=pp.create_folds(df,folds,label,shuffle,fold_col)
      df[processed_x_col]=pp.apply_pipeline(pipeline1,df[x_col])
      if save_file_path:
        df.to_csv(save_file_path,index=False)
      return df
    
  
if __name__=='__main__':
      # ! kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
      import os
      cur_dir=os.path.dirname(__file__)
      parent_dir=os.path.dirname(cur_dir)
      os.chdir(parent_dir)
      
      df=pd.read_csv(config.TRAIN_FILE)
      df=preprocess(df,
                 folds=5,
                 label=config.LABEL,x_col=config.TEXT_COL,
                 processed_x_col=config.PROC_TEXT_COL,
                 shuffle=True,
                 fold_col=config.KFOLD_COL,
                 save_file_path=config.PROCESSED_FILEPATH
                 )
      
      