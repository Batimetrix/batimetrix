import datetime
import earthaccess

earthaccess.login(strategy="environment")
end = datetime.date.today()

def latest(short_name, days, folder):
    start = end - datetime.timedelta(days=days)
    res = earthaccess.search_data(short_name=short_name, temporal=(start.isoformat(), end.isoformat()))
    if not res:
        print('No files for', short_name)
        return
    res = sorted(res, key=lambda g: g['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime'])
    print(short_name, '->', earthaccess.download(res[-1:], local_path=folder))

latest('NASA_SSH_REF_SIMPLE_GRID_V11', 30, './ssh_data')
latest('OSCAR_L4_OC_NRT_V2.0', 21, './oscar_data')
