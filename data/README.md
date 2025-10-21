# Data Directory

This folder contains all data files for the project. Only `.dvc` metafiles are tracked by git; actual data is stored in S3 and managed by DVC.

## What you’ll see after setup
```
/data
	/raw
		paysim-dev.csv
		paysim-dev.csv.dvc
		paysim-staging.csv
		paysim-staging.csv.dvc
		paysim-prod.csv
		paysim-prod.csv.dvc
		paysim_data.csv
		paysim_data.csv.dvc
	/processed
		processed_train.csv
		processed_test.csv
		train.csv
		test.csv
```

- `/raw`: Original data files for each environment (dev, staging, prod)
- `/processed`: Cleaned/engineered data for model training/testing
- `.dvc` files: Tracked by git/DVC; CSVs are pulled from S3

## How to get the data
Run:
```sh
dvc pull -r <DVC_REMOTE>
# <DVC_REMOTE> is dev, staging, or prod
```
See the full setup guide: [docs/setup/data.md](../docs/setup/data.md)