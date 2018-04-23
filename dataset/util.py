def download(url, filename):
    import sys
    if sys.version_info.major == 3:
        from urllib.request import urlopen
    else:
        from urllib2 import urlopen
    response = urlopen(url)
    chunk_size = 16 * 1024
    with open(filename, 'wb') as fout:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            fout.write(chunk)