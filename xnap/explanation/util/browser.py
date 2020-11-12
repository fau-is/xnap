import os
import webbrowser


def display_html(html):

    path = os.path.abspath('temp.html')
    url = 'file://' + path

    with open(path, 'w') as f:
        f.write(str(html))
    webbrowser.open(url)


