# huotarim-xgboost-li-ion-batteries -> relinked to # li-ion-battery-packs

The project huotarim/li-ion-battery-packs contains timesieries of six lithium-ion battery packs' state-of-health in six csv files.

The project is useful for those creating forecasts on battery packs.

Use the data e.g. with Python (Python 3.8 used in the original project).

The data is as is, i.e., there is no active updating of the data.

Contents of each csv-file: columns 1-15: X (i.e. the predictors), column 16: y (i.e. the target state-of-health)

X in more detail: columns 1-10 are the basic signals for a battery pack, columns 11-15 are the signals derived from SoH signal decomposition. See more details in the papers with my name on lithium-ion battery packs.
As a general recommendation for the first model building: drop columns 11-15 and use the rest.
