import pandas as pd


def _test_year_is_empty(in_df):
    return len(in_df[(in_df.year == '')]) != len(in_df)


def _test_valid_mapping(row):
    '''Must have fleet_id and device_id 
    or make and model
    or both'''

    if (row.fleet_id != '' and row.fleet_id is not None) \
            and (row.device_id != '' and row.device_id is not None):
        return 1

    if (row.make != '' and row.make is not None) \
            and (row.model != '' and row.model is not None):
        return 1

    return 0


def test_device_mapping(df):
    '''
    This is a linter for device mapping
    '''
    has_error = False
    errors = []

    if _test_year_is_empty(df):
        raise ValueError(f'Year column must be empty!')

    if any(df.duplicated()):
        raise ValueError('Duplicate Device Mappings!')

    for row in df.itertuples():
        if row.business_country == '' or row.business_country is None:
            has_error = True
            errors.append('business_country')
            print(ValueError(f'{row.Index} is missing business_country!'))

        if row.business_region == '' or row.business_region is None:
            has_error = True
            errors.append('business_region')
            print(ValueError(f'{row.Index} is missing business_region!'))

        if _test_valid_mapping(row) == 0:
            has_error = True
            errors.append('Make/Model or Fleet_id/Device_id')
            print(ValueError(f'INVALID COMBO: {row.Index}'))

    if has_error is True:
        raise ValueError(f'Reported Columns with Errors: {set(errors)}')

    return
