# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:30:45 2018

@author: Administrator
"""
def sqrloss(dst_value,content_value):
    return tf.reduce_mean(tf.square(dst_value-content_value))

layer = [256,128,63,3]

MEAN_VALUES = np.array([103.939, 116.779, 123.68]).reshape((1,1,3))

img_A = (A-MEAN_VALUES).astype('float32')
A_model = VGG19(img_A)
img_B = (B-MEAN_VALUES).astype('float32')
B_model = VGG19(img_B)

A_layers = A_model.forward_propogation()
A_feat = [A_layers['relu5_1'],A_layers['relu4_1'],A_layers['relu3_1'],A_layers['relu2_1'],A_layers['relu1_1']]


B_layers = B_model.forward_propogation()
B_feat = [B_layers['relu5_1'],B_layers['relu4_1'],B_layers['relu3_1'],B_layers['relu2_1'],B_layers['relu1_1']]

sess = tf.InteractiveSession()

Al_5 = A_feat[0].eval()
Al_5  = np.squeeze(Al_5)
At5 = np.clip(Al_5[:,:,2], 0,255)
cv2.imwrite('At5.jpg',At5)

Al_5_norm  = Al_5/np.sqrt(np.sum(Al_5**2,keepdims = True))
Alm_5_norm = Al_5_norm.copy()
Alm_5 = Al_5.copy()

Blm_5 = B_feat[0].eval()
Blm_5  = np.squeeze(Blm_5)
Bt5 = np.clip(Blm_5[:,:,2], 0,255)
cv2.imwrite('Bt5.jpg',Bt5)

Blm_5_norm  = Blm_5/np.sqrt(np.sum(Blm_5**2,keepdims = True))
Bl_5_norm = Blm_5_norm.copy()
Bl_5 = Blm_5.copy()

nnf_init_A = initialize_nnf(Al_5.shape[0:2],Bl_5.shape[0:2])
nnf_value_init_A = initialize_nnf_value(nnf_init_A,Al_5_norm,Alm_5_norm,Bl_5_norm,Blm_5_norm,1)

nnf_init_Bm = initialize_nnf(Bl_5.shape[0:2],Al_5.shape[0:2])
nnf_value_init_Bm = initialize_nnf_value(nnf_init_Bm,Blm_5_norm,Bl_5_norm,Alm_5_norm,Al_5_norm,1)

nnf_A,nnf_value_A = propogation(nnf_init_A,nnf_value_init_A,Al_5_norm,Alm_5_norm,Bl_5_norm,Blm_5_norm,1,4)            
warped_A = reconstruct(Al_5,Blm_5,nnf_A)
wAt5 = np.clip(warped_A[:,:,2], 0,255)
cv2.imwrite('wAt5.jpg',wAt5)

nnf_B,nnf_value_Bm = propogation(nnf_init_Bm,nnf_value_init_Bm,Blm_5_norm,Bl_5_norm,Alm_5_norm,Al_5_norm,1,4)            
warped_Bm = reconstruct(Blm_5,Al_5,nnf_B)
wBt5 = np.clip(warped_Bm[:,:,2], 0,255)
cv2.imwrite('wBt5.jpg',wBt5)

#'-----------------------'
with tf.variable_scope('L{}'.format(5),reuse = tf.AUTO_REUSE):
    Al_relu41  = tf.get_variable('Ra',shape = A_feat[1].get_shape().as_list()[0:-1] + [layer[0]],dtype = tf.float32,
                            initializer = tf.truncated_normal_initializer(stddev=1.0))
    Blm_relu41 = tf.get_variable('Rbm',shape = B_feat[1].get_shape().as_list()[0:-1] + [layer[0]],dtype = tf.float32,
                            initializer = tf.truncated_normal_initializer(stddev=1.0))

R_Al  = VGG19(Al_relu41).submodel(sub_id = 'fourth')
R_Blm = VGG19(Blm_relu41).submodel(sub_id = 'fourth')

loss_A  = sqrloss(warped_A, R_Al)
loss_Bm = sqrloss(warped_Bm, R_Blm)
optimizerA = tf.contrib.opt.ScipyOptimizerInterface(loss_A, method='L-BFGS-B', options={'maxiter': 600})
optimizerB = tf.contrib.opt.ScipyOptimizerInterface(loss_Bm, method='L-BFGS-B', options={'maxiter': 600})

holder1= tf.placeholder(tf.float32)
holder2= tf.placeholder(tf.float32)

tf.global_variables_initializer().run()

optimizerA.minimize(sess,feed_dict={holder1:warped_A})
optimizerB.minimize(sess,feed_dict={holder2:warped_Bm})



R_Al_4 = np.squeeze(VGG19(Al_relu41).trueresult(sub_id = 'fourth').eval())
RAt4 = np.clip(R_Al_4[:,:,2], 0,255)
cv2.imwrite('RAt4.jpg',RAt4)

Al_4 = A_feat[1].eval()
Al_4  = np.squeeze(Al_4)
At4 = np.clip(Al_4[:,:,2], 0,255)
cv2.imwrite('At4.jpg',At4)

R_Blm_4 = np.squeeze(VGG19(Blm_relu41).trueresult(sub_id = 'fourth').eval())
RBt4 = np.clip(R_Blm_4[:,:,2], 0,255)
cv2.imwrite('RBt4.jpg',RBt4)

Blm_4 = B_feat[1].eval()
Blm_4  = np.squeeze(Blm_4)
Bt4 = np.clip(Blm_4[:,:,2], 0,255)
cv2.imwrite('Bt4.jpg',Bt4)







