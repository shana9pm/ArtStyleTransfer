from utils import *
import tqdm
import time
from scipy.optimize import fmin_l_bfgs_b
from Settings import *
import keras.backend as K
import copy
import matplotlib.pyplot as plt

def calculate_loss_grad(outputImg):
    if outputImg.shape != (1, WIDTH, WIDTH, 3):
        outputImg = outputImg.reshape((1, WIDTH, HEIGHT, 3))
    loss=get_total_loss(outModel.input)
    grads=K.gradients(loss,[outModel.input])
    loss_fcn = K.function([outModel.input], [loss,grads[0]])
    f_out=loss_fcn([outputImg])
    return f_out[0].astype('float64'),f_out[1].flatten().astype('float64')


def get_total_loss(outputPlaceholder):
    F = get_feature_reps(outputPlaceholder,layer_names=[contentLayerNames], model=outModel)[0]
    Gs = get_feature_reps(outputPlaceholder,layer_names=styleLayerNames, model=outModel)
    contentLoss = get_content_loss(F, P)
    styleLoss = get_style_loss(ws, Gs, As)
    totalLoss = alpha*contentLoss + beta*styleLoss
    return totalLoss

def get_content_loss(F, P):
    if lossType=='SE':
        cLoss = 0.5*K.sum(K.square(F - P))
    else:
        cLoss= 0.5*K.sum(K.abs(F - P))
    return cLoss

def get_style_loss(ws, Gs, As):
    sLoss = K.variable(0.)
    for w, G, A in zip(ws, Gs, As):
        M_l = K.int_shape(G)[1]
        N_l = K.int_shape(G)[0]
        G_gram = get_Gram_matrix(G)
        A_gram = get_Gram_matrix(A)
        if lossType=='SE':
            sLoss+= w*0.25*K.sum(K.square(G_gram - A_gram))/ (N_l**2 * M_l**2)
        else:
            sLoss += w * 0.25 * K.sum(K.abs(G_gram - A_gram)) / (N_l ** 2 * M_l ** 2)
    return sLoss

def get_Gram_matrix(F):
    G = K.dot(F, K.transpose(F))
    return G


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




"""The following functions are basically used for loss record"""

def get_style_loss_forward(outputPlaceholder):
    Gs = get_feature_reps(outputPlaceholder, layer_names=styleLayerNames, model=outModel)
    styleLoss = get_style_loss(ws,Gs,As)
    return styleLoss

def get_content_loss_forward(outputPlaceholder):
    F = get_feature_reps(outputPlaceholder, layer_names=[contentLayerNames], model=outModel)[0]
    contentLoss = get_content_loss(F,P)
    return contentLoss

def calculate_style_loss(Xi):
    if Xi.shape != (1, WIDTH, WIDTH, 3):
        Xi = Xi.reshape((1, WIDTH, HEIGHT, 3))
    loss_fcn = K.function([outModel.input], [get_style_loss_forward(outModel.input)])
    return loss_fcn([Xi])[0].astype('float64')

def calculate_content_loss(Xi):
    if Xi.shape != (1, WIDTH, WIDTH, 3):
        Xi = Xi.reshape((1, WIDTH, HEIGHT, 3))
    loss_fcn = K.function([outModel.input], [get_content_loss_forward(outModel.input)])
    return loss_fcn([Xi])[0].astype('float64')

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = calculate_loss_grad(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()
if __name__=='__main__':

    parser=build_parser()
    args=parser.parse_args()
    content_name=args.content
    style_name=args.style
    output_name=args.output
    iteration=int(args.iter)
    flw=int(args.flw)
    lossType=args.losstype
    record=False if args.record=='F' else True
    rstep=int(args.rstep)
    stop = iteration // rstep
    alpha=float(args.alpha)
    beta=float(args.beta)
    fromc=False if args.fromc=='F' else True

    contentLossList=[]
    styleLossList=[]
    totalLossList=[]




    contentImgArr, contentOrignialImgSize = inputImageUtils(PATH_INPUT_CONTENT+content_name, SIZE)
    styleImgArr, styleOrignialImgSize = inputImageUtils(PATH_INPUT_STYLE+style_name, SIZE)
    if fromc:
        output, outputPlaceholder=outImageUtils2(PATH_INPUT_CONTENT+content_name,WIDTH,HEIGHT)
    else:
        output, outputPlaceholder = outImageUtils(WIDTH, HEIGHT)

    contentModel, styleModel, outModel = BuildModel(contentImgArr, styleImgArr, outputPlaceholder)

    P = get_feature_reps(x=contentImgArr,layer_names=[contentLayerNames], model=contentModel)[0]
    As = get_feature_reps(x=styleImgArr,layer_names=styleLayerNames, model=styleModel)
    ws = wlList[flw]


    outputImg=output.flatten()
    start=time.time
    count=tqdm.tqdm(total=iteration)
    name_list = output_name.split('.')
    for i in range(iteration):
        outputImg, f_val, info= fmin_l_bfgs_b(evaluator.loss, outputImg, fprime=evaluator.grads,maxfun=20)
        if record:
            deepCopy = copy.deepcopy(outputImg)
            this_styleLoss = calculate_style_loss(deepCopy)
            this_contentLoss = calculate_content_loss(deepCopy)
            contentLossList.append(this_contentLoss)
            styleLossList.append(this_styleLoss)
            this_totalLoss = this_styleLoss * beta + this_contentLoss * alpha
            totalLossList.append(this_totalLoss)
        if i%rstep==0:
            deepCopy = copy.deepcopy(outputImg)
            iter = i // rstep
            xOut = postprocess_array(deepCopy)
            imgName = PATH_OUTPUT + '.'.join(name_list[:-1]) + '_{}.{}'.format(
                str(iter) if iter != stop else 'final', name_list[-1])
            _ = save_original_size(xOut, imgName, contentOrignialImgSize)
    if record:
        plt.plot(totalLossList)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('TotalLoss')
        plt.savefig(PATH_OUTPUT + 'TotalLoss.jpg')

        plt.figure()
        plt.plot(contentLossList)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('ContentLoss')
        plt.savefig(PATH_OUTPUT + 'ContentLoss.jpg')

        plt.figure()
        plt.plot(styleLossList)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('StyleLoss')
        plt.savefig(PATH_OUTPUT + 'StyleLoss.jpg')
