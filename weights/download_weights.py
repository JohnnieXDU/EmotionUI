import urllib.request

print("Starting downloading weights ...")

url = "https://doc-14-5o-docs.googleusercontent.com/docs/securesc/re23ndjj0u6tqd602a7qj4qbsbnkrl69/aqu3p6ichb1d9ds0iagsbbebt9lvgrng/1639403400000/01281681549409338080/01281681549409338080/1f-tKgovJ54l9xR3oIZ6gy77NdPirUddr?e=download&authuser=0&nonce=q8vti9rbklnac&user=01281681549409338080&hash=vk0e9ia1efhbkid6cl272hnqabj467mj"

urllib.request.urlretrieve(url, "./vgg16_109lab.pth")

print("Done.")