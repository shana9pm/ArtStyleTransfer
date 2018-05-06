from utils import *
import tqdm
import time
from scipy.optimize import fmin_l_bfgs_b
from Settings import *
import keras.backend as K

def calculate_loss(outputImg):
    if outputImg.shape != (1, WIDTH, WIDTH, 3):
        outputImg = outputImg.reshape((1, WIDTH, HEIGHT, 3))
    loss_fcn = K.function([outModel.input], [get_total_loss(outModel.input)])
    return loss_fcn([outputImg])[0].astype('float64')

def get_total_loss(outputPlaceholder,alpha=1.0, beta=10000.0):
    F = get_feature_reps(outputPlaceholder,layer_names=[contentLayerName], model=outModel)[0]
    Gs = get_feature_reps(outputPlaceholder,layer_names=styleLayerNames, model=outModel)
    contentLoss = get_content_loss(F, P)
    styleLoss = get_style_loss(ws, Gs, As)
    totalLoss = alpha*contentLoss + beta*styleLoss
    return totalLoss

def get_content_loss(F, P):
    cLoss = 0.5*K.sum(K.square(F - P))
    return cLoss

def get_style_loss(ws, Gs, As):
    sLoss = K.variable(0.)
    for w, G, A in zip(ws, Gs, As):
        M_l = K.int_shape(G)[1]
        N_l = K.int_shape(G)[0]
        G_gram = get_Gram_matrix(G)
        A_gram = get_Gram_matrix(A)
        sLoss+= w*0.25*K.sum(K.square(G_gram - A_gram))/ (N_l**2 * M_l**2)
    return sLoss

def get_Gram_matrix(F):
    G = K.dot(F, K.transpose(F))
    return G

def get_grad(gImArr):
    """
    Calculate the gradient of the loss function with respect to the generated image
    """
    if gImArr.shape != (1, WIDTH,HEIGHT, 3):
        gImArr = gImArr.reshape((1, WIDTH,HEIGHT, 3))
    grad_fcn = K.function([outModel.input], K.gradients(get_total_loss(outModel.input), [outModel.input]))
    grad = grad_fcn([gImArr])[0].flatten().astype('float64')
    return grad

def get_feature_reps(x,layer_names, model):

    featMatrices = []
    for ln in layer_names:
        selectedLayer = model.get_layer(ln)
        featRaw = selectedLayer.output
        featRawShape = K.shape(featRaw).eval(session=K.get_session())
        N_l = featRawShape[-1]
        M_l = featRawShape[1]*featRawShape[2]
        featMatrix = K.reshape(featRaw, (M_l, N_l))
        featMatrix = K.transpose(featMatrix)
        featMatrices.append(featMatrix)
    return featMatrices

def callbackF(Xi):
    """A call back function for scipy optimization to record Xi each step"""
    global iterator
    global name_list
    global count
    if iterator%50==0:
        i = int(iterator / 50)
        xOut = postprocess_array(Xi)
        imgName = PATH_OUTPUT + '.'.join(name_list[:-1]) + '_{}.{}'.format(
            str(i + 1), name_list[-1])
        xIm = save_original_size(xOut, imgName, contentOrignialImgSize)

    iterator+=1
    count.update(1)

if __name__=='__main__':

    parser=build_parser()
    args=parser.parse_args()
    content_name=args.content
    style_name=args.style
    output_name=args.output
    iteration=int(args.iter)
    contentImgArr, contentOrignialImgSize = inputImageUtils(PATH_INPUT_CONTENT+content_name, SIZE)
    styleImgArr, styleOrignialImgSize = inputImageUtils(PATH_INPUT_STYLE+style_name, SIZE)
    output, outputPlaceholder = outImageUtils(WIDTH, HEIGHT)
    contentModel, styleModel, outModel = BuildModel(contentImgArr, styleImgArr, outputPlaceholder)

    P = get_feature_reps(x=contentImgArr,layer_names=[contentLayerName], model=contentModel)[0]
    As = get_feature_reps(x=styleImgArr,layer_names=styleLayerNames, model=styleModel)
    ws = np.ones(len(styleLayerNames))/float(len(styleLayerNames))


    outputImg=output.flatten()
    start=time.time
    count=tqdm.tqdm(total=iteration)
    name_list = output_name.split('.')
    iterator=0
    xopt, f_val, info= fmin_l_bfgs_b(calculate_loss, outputImg, fprime=get_grad,
                                maxiter=iteration, disp=False,callback=callbackF)

    xOut = postprocess_array(xopt)
    xIm = save_original_size(xOut, PATH_OUTPUT + output_name, contentOrignialImgSize)


