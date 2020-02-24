# Gradient-based Training and Pruning of Radial Basis Function Networks

## How to use RBF networks in your own work

The source file *rbfn.py* contains an independent implementation of
gradient-based radial basis function networks as a standard
PyTorch module. It also contains a class hierarchy of loss function
factories that can be used to prune RBF networks.

The file *train.py* contains more high-level methods for training
and pruning RBF networks. See *toy_example.py* or *traineval_task.py*
for examples of how to use them.

## How to reproduce the results of the paper

1.
    a. **(Using virtualenv)** Create a Python virtualenv and install the
    necessary packages:
    ````
   $ virtualenv -p `which python3` rbfn_venv
   $ source rbfn_venv/bin/activate
   $ pip install numpy pandas sklearn xgboost
    ````
    Also install PyTorch according to the instructions at
    <https://pytorch.org/>.

    b. **(Using Anaconda)** Create a conda virtual environment with all the
    required python and R packages. Activate the environment and run the scripts
    in it:
     ````
   $ conda env create env/rbfn_env.yml
   $ conda activate rbfn
     ````
2. Run the Python script *toy_example.py* to reproduce the toy example
   described in the paper.
3. If you are running the experiments on a cluster, make sure you have the
   copper dataset pre-downloaded:
   ````
   $ python -c "import util.dataset as d; _=d.load_cu_migration_barriers('100')"
   ````
4. Run the XGboost and DNN hyperparameter tuning tasks: Either modify the
   environment variables and Slurm module names in *xgb_tuning.job* and
   *dnn_tuning.job* according to your Slurm cluster environment and run them
   with *sbatch*, or manually execute the steps within if you do not have a
   cluster available.
5. Make sure the JSON files from the previous step are in the subdirectories
   *output/xgb_tuning_raw/* and *output/dnn_tuning_raw/*, then run the script
   *tuning_postprocess.py* twice, first with the command-line argument *xgb*,
   then with *dnn*, to combine the results.
6. Run the RBFN, XGBoost, and DNN training tasks with Slurm using the script
   *traineval.job*.
7. Similarly to the above, make sure the JSON and .model.gz files from the
   previous step are in the directory *output/traineval_raw*, then run
   *traineval_postprocess.py*.
8. Finally, run the R script *traineval_analysis.R* to reproduce the
   values in Table 1 of the paper.

## Citation

If you use this work in an academic publication,
please use the following citation:

````
@misc{rbfn,
  author={Jussi Määttä and Viacheslav Bazaliy and Jyri Kimari and Flyura Djurabekova and Kai Nordlund and Teemu Roos},
  title={Gradient-Based Training and Pruning of Radial Basis Function Networks with an Application in Materials Physics},
  year={2020},
  eprint={TODO},
  archivePrefix={arXiv},
  primaryClass={TODO}
}
````
