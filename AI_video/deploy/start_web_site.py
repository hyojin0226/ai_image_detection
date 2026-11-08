"""Open the fixed cloud website. No laptop server or Vercel login is needed."""
import argparse
import webbrowser

URL = 'https://duhyeon-ai-video.vercel.app'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if not args.no_browser:
        webbrowser.open(URL)
    print('온라인 서버에서 분석합니다. 이 프로그램이나 노트북을 켜 둘 필요가 없습니다.')

if __name__ == '__main__':
    main()
