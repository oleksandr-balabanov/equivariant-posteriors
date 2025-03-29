import experiments.weather.cdsmontly as cdstest

for year in range(2007, 2018):
    for month in range(1, 13):
        year_str = str(year)
        month_str = f"{month:02d}"

        try:
            e5sc = cdstest.ERA5SampleConfig(
                year=year_str, month=month_str, time="00:00:00"
            )
            e5s = cdstest.get_era5_sample(e5sc)
            e5_target_config = cdstest.ERA5SampleConfig(
                year=year_str, month=month_str, time="06:00:00"
            )
            e5target = cdstest.get_era5_sample(e5_target_config)
        except Exception as e:
            print(f"[Fail] {e}")
