import datetime
import earthaccess

earthaccess.login(strategy="environment")

end = datetime.date.today()
start = end - datetime.timedelta(days=30)
results = earthaccess.search_data(
    short_name='NASA_SSH_REF_SIMPLE_GRID_V11',
    temporal=(start.isoformat(), end.isoformat()),
)
if not results:
    raise SystemExit('No new SSH files found')
results = sorted(results, key=lambda g: g['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime'])
files = earthaccess.download(results[-1:], local_path='./ssh_data')
print('Latest file:', files)
