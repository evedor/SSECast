# Training and testing workflow

1. Prepare chronological and non-overlapping train, eval and test splits according to data/README.md. Compute normalization values from train only.
2. Select a regional configuration in configs/ and train one horizon:

   python scripts/train.py --config configs/cascadia.yaml --config-name backbone --run-name ssecast-14 --horizon 14 --device cuda --amp

3. The eval split selects best_model.ckpt. Training stores checkpoints, train_history.json and TensorBoard logs under outputs/<region>/<run-name>/.
4. Test the selected checkpoint once on the held-out split:

   python scripts/test.py --config configs/cascadia.yaml --config-name backbone --run-name ssecast-14 --horizon 14 --checkpoint outputs/cascadia/ssecast-14/training_checkpoints/best_model.ckpt --device cuda

5. Testing never changes the model. It writes ssecast_metrics.csv, ssecast_metrics.png and test_args.json. The CSV gives NRMSE and ACC for all three source-field components at every lead time.

Train separate models for 14 and 30 days; they are not recursive extensions of one another. Do not choose checkpoints, normalization constants or hyperparameters using test results.
