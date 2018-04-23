def download(url, filename):
    import sys
    version = sys.version_info
    if version > (2, 7):
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