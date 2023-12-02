# vits-Chat

----------------------------------------------------------------

# Streamlit infer for the vitsChats




- 두 가지 버전의 Streamlit 호출 스크립트입니다.

  1. vits webui version + OpenAI api key 입력 버전

  2. vits webui version + LM studio (local inference server) 버전

----------------------------------------------------------------

# 1. vits-Chat OAI

- 설치
  (기존에 vits가 설치가 되어 있다고 가정합니다.)

  1) 저장소 복제

    ```sh 
    git clone https://github.com/ilya-scherzo/vitsChats.git
    ```

  2) 저장소 파일을 기존 vits 설치 폴더로 복사합니다.
    
     recommend: https://github.com/ouor/vits

     vits 다른 버전은 테스트를 못했습니다.

  3) 종속성 설치를 위해 가상환경을 로드합니다.
     
     위 저장소 설치 기준
    
    ```sh 
    .\.venv\Scripts\activate
    ```

  4) 종속성 두 가지를 설치합니다.

    ```sh
    pip install -r requirements-vits-chats.txt
    ```


- 사전 학습모델 (checkpoint, config) 지정
  
  1) vits 추론을 위해 사전 학습모델이 필요합니다.
     
     기존 vits 저장소를 통해 학습한 사전 학습 모델의 경로를 다음 파일을 수정하여 지정합니다.

     ```
     model_path.json
     ```

  2) config 과 model 의 경로를 model_path.json 파일에 저장하면 자동으로 불러옵니다.

    ```
    example
    {
    "config_path": "ckpt/config.json",
    "model_path": "ckpt/G_1164.pth"
    }
    ```


- 실행
  
  1) 입력
  ```sh 
    streamlit run vits-chat-oai.py
  ```

  2) 자동으로 브라우저에 ui가 나옵니다.

![image](https://github.com/ilya-scherzo/vitsChats/assets/142293912/68eb45b6-3664-4158-bf26-166ded117f34)


  3) 자기가 선호하는 api model을 선택하고 자신의 api key를 입력합니다.

     (azure openai 사용자는 아래쪽 필드를 입력하면 되지만 테스트 해 보진 않았습니다.)

  4) set parameter를 누르고 정상이면 초록불이 들어옵니다.

![image](https://github.com/ilya-scherzo/vitsChats/assets/142293912/92bf38a2-e9bc-4a75-be57-89a7a695508c)


  5) 왼쪽의 choose a page를 누르면 settings page가 나옵니다.

  6) 프롬프트 입력 창과 vits 파라미터 조절창이 로드됩니다.

    ![image](https://github.com/ilya-scherzo/vitsChats/assets/142293912/1703c1b6-18ca-4ba3-bbbb-e33fc63ec164)


  7) 자신의 설정에 맞게 조절하시면 됩니다.

  8) 설정을 다 하셨으면 다시 왼쪽 choose a page에서 chat을 선택합니다.

  9) 채팅창이 나오고 채팅을 하면 됩니다.

      ![image](https://github.com/ilya-scherzo/vitsChats/assets/142293912/e45733ee-b5f5-4059-8c0f-46b50ed8e1be)


  10) api로 채팅을 하고 text를 로컬이 받아서 추론을 하는 방식이라 딜레이가 있습니다.

      openai의 tts가 더 자연스러워 지면 사장될 기술이기에 더 이상의 업뎃은 없을 예정입니다.


----------------------------------------------------------------

# 2. vits-Chat LM studio

- 설치, 사전학습 모델, 실행 등 거의 모든 사항이 위와 동일합니다.

  1) LM studio의 local inference server를 OAI api처럼 사용하여 양자화된 모델을 올려서 채팅을 합니다.

  2) LM studio의 사용 방법은 https://github.com/lmstudio-ai 를 참고하세요

  3) 로컬 서버를 기본 모드로 로드하면 서버 주소가 "http://localhost:1234/v1" 입니다.

  4) 기본적으로 스크립트에 입력되어 있기에 수정하실 필요는 없습니다.

  5) 세팅하고, 채팅하면 됩니다.

  6) 단, 로컬 장비 성능에 따라 답변, 음성 생성에 시간이 걸릴 수 있습니다.

  ----------------------------------------------------------------

 Reference
 1. https://github.com/jaywalnut310/vits
 2. https://github.com/ouor/vits
 3. https://github.com/lmstudio-ai
 4. https://platform.openai.com/docs/guides/text-generation/chat-completions-api
 5. https://docs.streamlit.io/library/api-reference

