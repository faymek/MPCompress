# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import enum  
import struct  
from pathlib import Path 
import torch

def filesize(filepath: str) -> int:                   #返回filepath的文件大小（字节数）
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size

def write_uints(fd, values, fmt=">{:d}I"):             # >大端序  unsigned int 4 Bytes  values：可迭代的整数列表/元组
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4

def read_uints(fd, n, fmt=">{:d}I"):                   # fd 文件 n读取的unsigned int的数量 返回(x,x,x)
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def write_uchars(fd, values, fmt=">{:d}B"):            # unsigned char 1 Bytes
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values)

def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def write_hex_bytes(fd, data: bytes):
    fd.write(data)
    return len(data)

def read_hex_bytes(fd, n):
    return fd.read(n)


def write_bytes(fd, values, fmt=">{:d}s"):    #str struct.pack(">5s", b"hello") → b"hello"  5Bytes
    if len(values) == 0:
        return 0
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values)


def read_bytes(fd, n, fmt=">{:d}s"):          #return str
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def write_ushorts(fd, values, fmt=">{:d}H"):  #unsigned short 2Bytes
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 2


def read_ushorts(fd, n, fmt=">{:d}H"):
    sz = struct.calcsize("H")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_uint_adaptive(f, a):  #根据数值大小动态选择1/2/4字节存储，通过最高位标记长度
    if a < (1 << 7):            #0xxxxxxx：1Bytes（0-127）
        a0 = (a >> 0) & 0xff
        a0 = a0 | (0x00 << 7)
        write_uchars(f, (a0,))
        return 1

    if a < (1 << 14):           #10xxxxxx xxxxxxxx：2Bytes（128-16383）
        a0 = (a >> 0) & 0xff
        a1 = (a >> 8) & 0xff
        a1 = a1 | (0x02 << 6)
        write_uchars(f, (a1, a0))
        return 2

    assert a < (1 << 30)       #11xxxxxx xxxxxxxx xxxxxxxx xxxxxxxx：4Bytes（16384-2^30）
    a0 = (a >> 0) & 0xff
    a1 = (a >> 8) & 0xff
    a2 = (a >> 16) & 0xff
    a3 = (a >> 24) & 0xff
    a3 = a3 | (0x03 << 6)
    write_uchars(f, (a3, a2, a1, a0))
    return 4


def read_uint_adaptive(f):
    a3 = read_uchars(f, 1)[0]
    if (a3 >> 7) == 0:
        return a3

    a2 = read_uchars(f, 1)[0]

    if (a3 >> 6) == 0x02:
        a3 = a3 & 0x3f
        return (a3 << 8) + a2
    a3 = a3 & 0x3f
    a1 = read_uchars(f, 1)[0]
    a0 = read_uchars(f, 1)[0]
    return (a3 << 24) + (a2 << 16) + (a1 << 8) + a0


class Start_Code(enum.Enum):
    SC_GLOBAL = b"\x00\x00\x01\x80"
    SC_IBRANCH1 = b"\x00\x00\x01\x81"
    SC_IBRANCH2 = b"\x00\x00\x01\x82"
    SC_IBRANCH3 = b"\x00\x00\x01\x83"
    SC_SEI = b"\x00\x00\x01\x84"
    SC_OBJ = b"\x00\x00\x01\x85"


PATCH_SIZE = {
    8: 0,
    14: 1,
    16: 2,
    32: 3,
    64: 4
}

IBRANCH1_TYPE ={
    "vqgan":0
}

IBRANCH2_TYPE ={
    "dino":0
}

IBRANCH3_TYPE ={}

def write_flag_byte(layer_outputs):  #sei | obj | ib1 | ib2 | ib3 | / | / | 1
    keys = layer_outputs.keys()
    
    sei_flag=1 if "sei" in keys else 0
    obj_flag=1 if "obj" in keys else 0
    ibranch1_flag=1 if any(k in keys for k in IBRANCH1_TYPE.keys()) else 0
    ibranch2_flag=1 if any(k in keys for k in IBRANCH2_TYPE.keys()) else 0
    ibranch3_flag=1 if any(k in keys for k in IBRANCH3_TYPE.keys()) else 0

    #sei_flag, obj_flag, ibranch1_flag, ibranch2_flag = (int(k in keys) for k in ('sei', 'obj', 'vqgan', 'dino'))  #可扩展
    
    flag_byte = sei_flag<<7 | obj_flag<<6 | ibranch1_flag<<5 | ibranch2_flag<<4 | 1

    return  flag_byte , (sei_flag, obj_flag, ibranch1_flag, ibranch2_flag,ibranch3_flag,False,False,1)

def read_flag_byte(flag_byte):
    sei_flag = bool(flag_byte & 0b10000000)
    obj_flag = bool(flag_byte & 0b01000000)
    ibranch1_flag = bool(flag_byte & 0b00100000)
    ibranch2_flag = bool(flag_byte & 0b00010000)
    ibranch3_flag = bool(flag_byte & 0b00001000)
    marker = bool(flag_byte & 0b00000001)
    
    return sei_flag, obj_flag, ibranch1_flag, ibranch2_flag,ibranch3_flag,False,False,marker


def write_extension_data(f,extension_flag,extension_data,data_length_bit):
    extension_written=0
    if extension_flag:
        length = len(extension_data)

        if data_length_bit==15:
            extension_written += write_ushorts(f,( 1 << 15 | length,))
        elif data_length_bit==7:
            extension_written += write_uchars(f,( 1 << 7 | length,))
        else:
            raise(ValueError(f'Something Wrong'))

        extension_written += write_uchars(f,extension_data)
        return extension_written
    else:
        extension_written += write_uchars(f,(0,))
        return extension_written


def read_extension_data(f,data_length_bit):
    extension = read_uchars(f,1)[0]   
    if extension & 0b10000000:       # 最高位是 1
        if data_length_bit==15:
            len_bytes = read_uchars(f,1)[0]   
            N = (extension  << 8 | len_bytes ) & 0x7FFF  # 低 15 位
        elif data_length_bit==7:
            N = extension & 0x7F  # 低 7 位
        else:
            raise(ValueError(f'Something Wrong'))
        return True,read_uchars(f,N)
    else:
        return False,None  # 最高位是 0


def write_stream(layer_outputs, bin_path):
    with Path(bin_path).open("wb") as f:
        written=0
        
        ################  global header  #############################
        written += write_bytes(f,Start_Code.SC_GLOBAL.value)
        written += write_ushorts(f,(layer_outputs["meta"]["shape"][1],layer_outputs["meta"]["shape"][2]))
        written += write_uchars(f,(layer_outputs["meta"]["pad"],))

        flag_byte,(sei_flag, obj_flag, ibranch1_flag, ibranch2_flag,ibranch3_flag,_,_,marker) = write_flag_byte(layer_outputs)
        written +=write_uchars(f,(flag_byte,))

        gh_extension_flag = 0  #暂定为0，写法后续可根据layer_outputs结构修改
        gh_extension_data = None
        #gh_extension_data = (3,4,5,8,203)
        written += write_extension_data(f,gh_extension_flag,gh_extension_data,15)

        ########################  sei  #############################
        if sei_flag:
            written += write_bytes(f,Start_Code.SC_SEI.value)
            written += write_uint_adaptive(f, len(layer_outputs["sei"]))
            written += write_bytes(f, layer_outputs["sei"].encode())

        ########################  obj  #############################
        if obj_flag:
            written += write_bytes(f,Start_Code.SC_OBJ.value)
            written += write_ushorts(f, len(layer_outputs["obj_data"]))
            written += write_uchars(f, layer_outputs["obj_data"])

        ########################  ibranch1  #########################
        if ibranch1_flag:
            written += write_bytes(f,Start_Code.SC_IBRANCH1.value)

            ibranch1_type = IBRANCH1_TYPE["vqgan"]  # 后续更改为layer_outputs存储的模型 e.g. layer_outputs["ibranch1"]["type"]
            
            height , width = layer_outputs["meta"]["shape"][1],layer_outputs["meta"]["shape"][2]
            height_num,width_num = layer_outputs["vqgan"]["shape"][0],layer_outputs["vqgan"]["shape"][1]
            pad = layer_outputs["meta"]["pad"]
            assert (height + pad - 1) // pad * pad //height_num == (width + pad - 1) // pad * pad //width_num
            patch_size = (height + pad - 1) // pad * pad //height_num  #这样计算有点麻烦？后续可将patch_size写入layer_outputs

            patch_size_id = PATCH_SIZE[patch_size]
            written += write_uchars(f, (ibranch1_type <<4  | patch_size_id,))
            written += write_uchars(f, (layer_outputs["vqgan"]["alphabet_size"].bit_length() - 11 <<4 ,))

            written += write_uint_adaptive(f, len(layer_outputs["vqgan"]["strings"][0]))
            written += write_hex_bytes(f, layer_outputs["vqgan"]["strings"][0])


            ibranch1_extension_flag = 0  #暂定为0，写法后续可根据layer_outputs结构修改
            ibranch1_extension_data=None
            #ibranch1_extension_data=(3,4,5,8,203)
            written += write_extension_data(f,ibranch1_extension_flag,ibranch1_extension_data,15)
            
        ########################  ibranch2  ######################
        if ibranch2_flag:
            written += write_bytes(f,Start_Code.SC_IBRANCH2.value)

            ibranch2_type = IBRANCH2_TYPE["dino"]  # 后续更改为layer_outputs存储的模型 e.g. layer_outputs["ibranch2"]["type"]
            
            height , width = layer_outputs["meta"]["shape"][1],layer_outputs["meta"]["shape"][2]
            height_num,width_num = layer_outputs["dino"]["token_res"][0],layer_outputs["dino"]["token_res"][1]
            pad = layer_outputs["meta"]["pad"]
            assert (height + pad - 1) // pad * pad //height_num == (width + pad - 1) // pad * pad //width_num
            patch_size = (height + pad - 1) // pad * pad //height_num  #后续可将patch_size写入layer_outputs

            patch_size_id = PATCH_SIZE[patch_size]
            written += write_uchars(f, (ibranch1_type <<4  | patch_size_id,))

            num_layers=len(layer_outputs["dino"]["strings"])
            written += write_uchars(f, (num_layers,))

            
            for i in range(num_layers):
                written += write_ushorts(f, (len(layer_outputs["dino"]["strings"][i][0]),))
                written += write_hex_bytes(f, layer_outputs["dino"]["strings"][i][0])

            ibranch2_extension_flag = 0  #暂定为0，写法后续可根据layer_outputs结构修改
            ibranch2_extension_data=None
            #ibranch2_extension_data=(3,4,5,8,203)
            written += write_extension_data(f,ibranch2_extension_flag,ibranch2_extension_data,15)

        ########################  ibranch3  #####################
        if ibranch3_flag:
            pass

    return written
 

def read_stream(bin_path):
    print_sig=0
    layer_outputs={}
    with Path(bin_path).open("rb") as f:
        global_start_code = read_bytes(f, 4)
        if global_start_code!=Start_Code.SC_GLOBAL.value:
            raise(ValueError(f'Unmatched start code for global head.'))

        layer_outputs["meta"]={}
        img_height,img_width = read_ushorts(f,2)
        layer_outputs["meta"]["shape"]=(3,img_height,img_width)
        pad=read_uchars(f,1)[0]
        layer_outputs["meta"]["pad"]=pad

        flag_byte = read_uchars(f,1)[0]
        sei_flag, obj_flag, ibranch1_flag, ibranch2_flag,ibranch3_flag,_,_,marker = read_flag_byte(flag_byte)
        if not marker:
            raise(ValueError(f'Something Wrong.'))

        gh_extension_flag , gh_extension_data = read_extension_data(f,15)

        if print_sig:
            print("-----global info-------")
            print("img_height,img_width:",img_height,img_width)
            print("pad:",pad)
            print("sei_flag:",sei_flag)
            print("obj_flag:",obj_flag)
            print("ibranch1_flag:",ibranch1_flag)
            print("ibranch2_flag:",ibranch2_flag)
            print("ibranch3_flag:",ibranch3_flag)
            print("gh_extension_flag:",gh_extension_flag," ","gh_extension_data:",gh_extension_data)

        if sei_flag:
            sei_start_code = read_bytes(f, 4)
            if sei_start_code !=Start_Code.SC_SEI.value:
                raise(ValueError(f'Unmatched start code for SEI.'))
            sei_data_length=read_uint_adaptive(f)
            layer_outputs["sei"] = read_bytes(f,sei_data_length).decode()

            if print_sig:
                print("-----sei info-------")
                print("sei_data_length:",sei_data_length)
                print("sei_data:",layer_outputs["sei"])

        if obj_flag:
            obj_start_code = read_bytes(f, 4)
            if obj_start_code !=Start_Code.SC_OBJ.value:
                raise(ValueError(f'Unmatched start code for object.'))
            obj_data_length = read_ushorts(f,1)
            layer_outputs["obj_data"]=obj_data=read_uchars(f,obj_data_length)
            if print_sig:
                print("-----object info-------")
                print("obj_data_length:",obj_data_length)
                print("obj_data:",layer_outputs["obj_data"])

        if ibranch1_flag:
            ibranch1_start_code = read_bytes(f, 4)
            if ibranch1_start_code !=Start_Code.SC_IBRANCH1.value:
                raise(ValueError(f'Unmatched start code for ibranch1.'))

            temp = read_uchars(f,1)[0]
            ibranch1_type = next(k for k, v in IBRANCH1_TYPE.items() if v == temp >> 4)
            layer_outputs[ibranch1_type]={}

            patch_size=next(k for k, v in PATCH_SIZE.items() if v == temp & 0x00FF)
            layer_outputs[ibranch1_type]['alphabet_size'] = 2**((read_uchars(f,1)[0] >> 4) + 10)
             
            layer_outputs[ibranch1_type]['shape']=((img_height + pad - 1) // pad * pad//patch_size,(img_width + pad - 1) // pad * pad//patch_size)
            
            vqgan_data_length=read_uint_adaptive(f)
            layer_outputs[ibranch1_type]['strings']=[read_hex_bytes(f,vqgan_data_length)]

            ibranch1_extension_flag , ibranch1_extension_data = read_extension_data(f,15)

            if print_sig:
                print("-----ibranch1 info-------")
                print("ibranch1_type:",ibranch1_type )
                print("patch_size:", patch_size )
                print("alphabet_size:", layer_outputs[ibranch1_type]['alphabet_size'])
                print("shape:",layer_outputs[ibranch1_type]['shape'])
                print("vqgan_data_length:",vqgan_data_length)
                #print("vqgan_data:",bool(vqgan_data))
                print("ibranch1_extension_flag:",ibranch1_extension_flag," ","ibranch1_extension_data:",ibranch1_extension_data)

        if ibranch2_flag:
            ibranch2_start_code = read_bytes(f, 4)
            if ibranch2_start_code !=Start_Code.SC_IBRANCH2.value:
                raise(ValueError(f'Unmatched start code for ibranch2.'))

            temp = read_uchars(f,1)[0]
            ibranch2_type = next(k for k, v in IBRANCH2_TYPE.items() if v == temp >> 4)
            layer_outputs[ibranch2_type]={}

            patch_size = next(k for k, v in PATCH_SIZE.items() if v == temp & 0x00FF)
            num_layers=read_uchars(f,1)[0]
            layer_outputs[ibranch2_type]["token_res"]=((img_height + pad - 1) // pad * pad//patch_size,(img_width + pad - 1) // pad * pad//patch_size)

            ####### dino shape的确定 是否需要传入某些配置参数 如 M 和 groups ########
            M=120
            groups=5
            layer_outputs[ibranch2_type]["shape"]={}
            dino_y=[]
            h,w=layer_outputs[ibranch2_type]["token_res"][0],layer_outputs[ibranch2_type]["token_res"][1]
            for i in range(M//groups):
                dino_y.append(torch.Size([groups,h//2,w//2]))
            layer_outputs[ibranch2_type]["shape"]["y"]=dino_y
            layer_outputs[ibranch2_type]["shape"]["hyper"]=torch.Size([h//8,w//8])
            
            dino_data=[]
            for i in range(num_layers):
                dino_layer_data_length=read_ushorts(f,1)[0]
                dino_data.append([read_hex_bytes(f,dino_layer_data_length)])

            layer_outputs[ibranch2_type]["strings"]=dino_data
            ibranch2_extension_flag , ibranch2_extension_data = read_extension_data(f,15)

            if print_sig:
                print("-----ibranch2 info-------")
                print("ibranch2_type:",ibranch2_type )
                print("patch_size:", patch_size)
                print("token_res:",layer_outputs[ibranch2_type]["token_res"])
                print("num_layers:",num_layers)
                print("ibranch2_extension_flag:",ibranch2_extension_flag," ","ibranch2_extension_data:",ibranch2_extension_data)

        if ibranch3_flag:
            ibranch3_start_code = read_bytes(f, 4)
            if ibranch3_start_code !=Start_Code.SC_IBRANCH3.value:
                raise(ValueError(f'Unmatched start code for ibranch3.'))
             
    return layer_outputs


if __name__ == "__main__":
    import pickle
    
    """
    说明：外部调用
    write_stream(layer_outputs, bin_path) 将layer_outputs内容写入 bin_path.bin  返回码流总字节长度
    read_stream(bin_path) 读取bin_path.bin   返回layer_outputs
    """

    ### test ###
    root = Path('./eval_mpc_samples/')  
    output_path='/home/LingQY/network/MPC/MPCompress/out.bin'     
    for pkl_path in root.rglob('*.pkl'):  
        input_path= pkl_path
        print("input_path:",input_path.name)
        with open(input_path, 'rb') as g:
            layer_out = pickle.load(g)
        length = write_stream(layer_out,output_path)
        read_out=read_stream(output_path)
        print("stream length:",length,"Bytes")
        
        if layer_out==read_out:
            print("Right")
        else:
            print("Wrong")



