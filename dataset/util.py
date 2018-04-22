def download(url, filename):
    from urllib.request import urlopen
    response = urlopen(url)
    chunk_size = 16 * 1024
    with open(filename, 'wb') as fout:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            fout.write(chunk)