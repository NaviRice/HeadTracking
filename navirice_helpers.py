

def navirice_img_set_write_file(session_name, img_set, last_count):
    name = 'DATA/' + session_name + "_" + str(last_count) + ".img_set"
    name = name.replace('\n', '')
    print("Recording to: ", name)
    f = open(name, 'wb')
    f.write(img_set.SerializeToString())
    f.close()

