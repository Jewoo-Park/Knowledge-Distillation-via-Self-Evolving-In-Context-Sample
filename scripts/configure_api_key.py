"""Run interactively in the server terminal; performs no network requests."""
import getpass
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.api_credentials import KEY_FILE, save_key


def main():
    if KEY_FILE.exists():
        print('이미 키 파일이 있습니다. 설정을 덮어쓰지 않았습니다.')
        return 1
    print('발급받은 OpenAI API 키를 붙여넣고 Enter를 누르세요. 입력 내용은 표시되지 않습니다.')
    try:
        with warnings.catch_warnings():
            # Never fall back to echoing the secret when no terminal is available.
            warnings.simplefilter('error', getpass.GetPassWarning)
            key = getpass.getpass('API key: ')
        save_key(key)
    except (getpass.GetPassWarning, EOFError, KeyboardInterrupt):
        print('\n입력을 중단했습니다. 서버의 대화형 터미널에서 실행해 주세요.')
        return 1
    except (ValueError, FileExistsError):
        print('저장하지 못했습니다. 키에 공백이 있는지 또는 키 파일이 이미 있는지 확인하세요.')
        return 1
    print('키 저장 완료. 파일 권한은 600입니다. API 호출은 아직 하지 않았습니다.')
    print('이제 채팅에 "설정 완료"라고 알려 주세요. 키 값은 보내지 마세요.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
