#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit, modules
from optparse import OptionParser

import tensorflow as tf
import numpy as np
import scipy

import math



# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True
argv.remove( '-b-' )

from array import array

from HiggsAnalysis.CombinedLimit.DatacardParser import *
from HiggsAnalysis.CombinedLimit.ModelTools import *
from HiggsAnalysis.CombinedLimit.ShapeTools import *
from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.tfscipyhess import ScipyHessOptimizerInterface
from HiggsAnalysis.CombinedLimit.bfgscustom import minimize_bfgs_custom, _minimize_trustregion_constr_custom


parser = OptionParser(usage="usage: %prog [options] datacard.txt -o output \nrun with --help to get list of options")
addDatacardParserOptions(parser)
parser.add_option("-P", "--physics-model", dest="physModel", default="HiggsAnalysis.CombinedLimit.PhysicsModel:defaultModel",  type="string", help="Physics model to use. It should be in the form (module name):(object name)")
parser.add_option("--PO", "--physics-option", dest="physOpt", default=[],  type="string", action="append", help="Pass a given option to the physics model (can specify multiple times)")
parser.add_option("", "--dump-datacard", dest="dumpCard", default=False, action='store_true',  help="Print to screen the DataCard as a python config and exit")
parser.add_option("-t","--toys", default=0, type=int, help="run a given number of toys, 0 fits the data (default), and -1 fits the asimov toy")
parser.add_option("","--toysFrequentist", default=True, action='store_true', help="run frequentist-type toys by randomizing constraint minima")
parser.add_option("","--bypassFrequentistFit", default=True, action='store_true', help="bypass fit to data when running frequentist toys to get toys based on prefit expectations")
parser.add_option("","--bootstrapData", default=False, action='store_true', help="throw toys directly from observed data counts rather than expectation from templates")
parser.add_option("","--tolerance", default=1e-5, type=float, help="convergence tolerance for minimizer")
parser.add_option("","--expectSignal", default=1., type=float, help="rate multiplier for signal expectation (used for fit starting values and for toys)")
parser.add_option("","--seed", default=123456789, type=int, help="random seed for toys")
(options, args) = parser.parse_args()

if len(args) == 0:
    parser.print_usage()
    exit(1)
    
seed = options.seed
print(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

options.fileName = args[0]
if options.fileName.endswith(".gz"):
    import gzip
    file = gzip.open(options.fileName, "rb")
    options.fileName = options.fileName[:-3]
else:
    file = open(options.fileName, "r")

## Parse text file 
DC = parseCard(file, options)

if options.dumpCard:
    DC.print_structure()
    exit()

print(options)

nproc = len(DC.processes)
nsyst = len(DC.systs)
npoi = len(DC.signals)

dtype = 'float64'

MB = ShapeBuilder(DC, options)

#determine number of bins for each channel
nbinschan = {}
nbinstotal = 0
for chan in DC.bins:
  expchan = DC.exp[chan]
  for proc in DC.processes:
    if proc in expchan:
      datahist = MB.getShape(chan,"data_obs")
      nbins = datahist.GetNbinsX()
      nbinschan[chan] = nbins
      nbinstotal += nbins
      break

#fill data, expected yields, and kappas

#n.b data and expected have shape [nbins]

#norm has shape [nbins,nprocs] and keeps track of expected normalization

#logkup/down have shape [nbins, nprocs, nsyst] and keep track of systematic variations
#per-bin, per-process, per nuisance-parameter 

logkepsilon = math.log(1e-3)

data_obs = np.empty([0],dtype=dtype)
norm = np.empty([0,nproc],dtype=dtype)
logkup = np.empty([0,nproc,nsyst],dtype=dtype)
logkdown = np.empty([0,nproc,nsyst],dtype=dtype)
for chan in DC.bins:
  expchan = DC.exp[chan]
  nbins = nbinschan[chan]
  
  datahist = MB.getShape(chan,"data_obs")
  datanp = np.array(datahist)[1:-1]
  data_obs = np.concatenate((data_obs,datanp))
  
  normchan = np.empty([nbins,0],dtype=dtype)
  logkupchan = np.empty([nbins,0,nsyst],dtype=dtype)
  logkdownchan = np.empty([nbins,0,nsyst],dtype=dtype)
  for proc in DC.processes:
    hasproc = proc in DC.exp[chan]
    
    if hasproc:
      normhist = MB.getShape(chan,proc)
      normnp = np.array(normhist).astype(dtype)[1:-1]
      normnp = np.reshape(normnp,[-1,1])
    else:
      normnp = np.zeros([nbins,1],dtype=dtype)
      
    normchan = np.concatenate((normchan,normnp),axis=1)
      
    logkupproc = np.empty([nbins,1,0],dtype=dtype)
    logkdownproc = np.empty([nbins,1,0],dtype=dtype)
    for syst in DC.systs:
      name = syst[0]
      stype = syst[2]
      
      if not hasproc:
        logkupsyst = np.zeros([nbins,1,1],dtype=dtype)
        logkdownsyst = np.zeros([nbins,1,1],dtype=dtype)
      elif stype=='lnN':
        ksyst = syst[4][chan][proc]
        if type(ksyst) is list:
          ksystup = ksyst[1]
          ksystdown = ksyst[0]
          if ksystup == 0.:
            ksystup = 1.
          if ksystdown == 0.:
            ksystdown = 1.
          logkupsyst = math.log(ksystup)*np.ones([nbins,1,1],dtype=dtype)
          logkdownsyst = -math.log(ksystdown)*np.ones([nbins,1,1],dtype=dtype)
        else:
          if ksyst == 0.:
            ksyst = 1.
          logkupsyst = math.log(ksyst)*np.ones([nbins,1,1],dtype=dtype)
          logkdownsyst = math.log(ksyst)*np.ones([nbins,1,1],dtype=dtype)
        
      elif 'shape' in stype:
        kfac = syst[4][chan][proc]
        
        if kfac>0:
          normhistup = MB.getShape(chan,proc,name+"Up")
          normnpup = np.array(normhistup).astype(dtype)[1:-1]
          normnpup = np.reshape(normnpup,[-1,1])
          logkupsyst = kfac*np.log(normnpup/normnp)
          if np.any(np.logical_and(np.equal(normnpup,0.),np.not_equal(normnp,0.))):
            print(['up',chan,proc,name])
          logkupsyst = np.where(np.equal(normnpup,0.),logkepsilon*np.ones_like(logkupsyst),logkupsyst)
          #logkupsyst = np.where(np.equal(normnpup,0.),np.zeros_like(logkupsyst),logkupsyst)
          logkupsyst = np.where(np.equal(normnp,0.),np.zeros_like(logkupsyst),logkupsyst)
          logkupsyst = np.reshape(logkupsyst,[-1,1,1])
          
          normhistdown = MB.getShape(chan,proc,name+"Down")
          normnpdown = np.array(normhistdown).astype(dtype)[1:-1]
          normnpdown = np.reshape(normnpdown,[-1,1])
          logkdownsyst = -kfac*np.log(normnpdown/normnp)
          if np.any(np.logical_and(np.equal(normnpdown,0.),np.not_equal(normnp,0.))):
            print(['down',chan,proc,name])
          logkdownsyst = np.where(np.equal(normnpdown,0.),-logkepsilon*np.ones_like(logkdownsyst),logkdownsyst)
          #logkdownsyst = np.where(np.equal(normnpdown,0.),np.zeros_like(logkdownsyst),logkdownsyst)
          logkdownsyst = np.where(np.equal(normnp,0.),np.zeros_like(logkdownsyst),logkdownsyst)
          logkdownsyst = np.reshape(logkdownsyst,[-1,1,1])
        else:
          logkupsyst = np.zeros([normnp.shape[0],1,1],dtype=dtype)
          logkdownsyst = np.zeros([normnp.shape[0],1,1],dtype=dtype)
      else:
        raise Exception('Unsupported systematic type')

      logkupproc = np.concatenate((logkupproc,logkupsyst),axis=2)
      logkdownproc = np.concatenate((logkdownproc,logkdownsyst),axis=2)  

    logkupchan = np.concatenate((logkupchan,logkupproc),axis=1)
    logkdownchan = np.concatenate((logkdownchan,logkdownproc),axis=1)  
    
  norm = np.concatenate((norm,normchan), axis=0)

  logkup = np.concatenate((logkup,logkupchan),axis=0)
  logkdown = np.concatenate((logkdown,logkdownchan),axis=0)
  
  
print(np.max(np.abs(logkup)))
print(np.max(np.abs(logkdown)))
  
logkavg = 0.5*(logkup+logkdown)
logkhalfdiff = 0.5*(logkup-logkdown)

#list of signals preserving datacard order
signals = []
for proc in DC.processes:
  if DC.isSignal[proc]:
    signals.append(proc)

#build matrix of signal strength effects
#hard-coded for now as one signal strength multiplier
#per signal process
kr = np.zeros([nproc,npoi],dtype=dtype)
for ipoi,signal in enumerate(signals):
  iproc = DC.processes.index(signal)
  kr[iproc][ipoi] = 1.

#initial value for signal strenghts
#rv = options.expectSignal*np.ones([npoi]).astype(dtype)
rv = math.log(options.expectSignal) + np.zeros([npoi]).astype(dtype)


#initial value for nuisances
thetav = np.zeros([nsyst]).astype(dtype)

#combined initializer for all fit parameters
rthetav = np.concatenate((rv,thetav),axis=0)


#data
#nobs = tf.placeholder(dtype, shape=data_obs.shape)
nobs = tf.Variable(data_obs, trainable=False)
theta0 = tf.Variable(np.zeros_like(thetav), trainable=False)


#tf variable containing all fit parameters
rtheta = tf.Variable(rthetav)
rotation = tf.Variable(np.eye(npoi+nsyst,dtype=dtype),trainable=False)
rthetatrans = tf.reshape(tf.matmul(rotation,tf.reshape(rtheta,[-1,1])),[-1])

#split back into signal strengths and nuisances
r = rtheta[:npoi]
theta = rtheta[npoi:]
#r = rthetatrans[:npoi]
#theta = rthetatrans[npoi:]

#matrices encoding effect of signal strengths
#rkr = tf.pow(r,kr)
rkr = tf.exp(r*kr)
rnorm = tf.reduce_prod(rkr, axis=-1)

#interpolation for asymmetric log-normal
twox = 2.*theta
twox2 = twox*twox
alpha =  0.125 * twox * (twox2 * (3*twox2 - 10.) + 15.)
alpha = tf.clip_by_value(alpha,-1.,1.)
logk = logkavg + alpha*logkhalfdiff

#matrix encoding effect of nuisance parameters
snorm = tf.reduce_prod(tf.exp(logk*theta),axis=-1)

#final expected yields per-bin including effect of signal
#strengths and nuisance parmeters
pnorm = snorm*rnorm*norm
#pnorm = tf.maximum(pnorm,tf.zeros_like(pnorm))
nexp = tf.reduce_sum(pnorm,axis=-1)
nexp = tf.identity(nexp,name='nexp')

nexpsafe = tf.where(tf.equal(nobs,tf.zeros_like(nobs)), tf.ones_like(nobs), nexp)
lognexp = tf.log(nexpsafe)

#final likelihood computation

#poison term
ln = tf.reduce_sum(-nobs*lognexp + nexp, axis=-1)

#constraints
lc = tf.reduce_sum(0.5*tf.square(theta - theta0))

l = ln + lc
l = tf.identity(l,name="loss")

grads = tf.gradients(l,rtheta)
grads = tf.identity(grads,"loss_grads")

grad = grads[0]

#uncertainty computation
hess = tf.hessians(l,rtheta)[0]
hess = tf.identity(hess,name="loss_hessian")

#unstackedgrad = tf.unstack(tf.reshape(grad,[1,-1]),axis=-1)
#gunstackedgrad = tf.gradients(unstackedgrad,rtheta)

#gradsq = tf.map_fn(lambda x : tf.gradients(x,rtheta)[0],grad,parallel_iterations=32)

#hessalt = tf.stack([tf.gradients(g,rtheta)[0] for g in tf.unstack(grad)])

#hessalt = tf.map_fn(lambda x: tf.gradients(x,rtheta),grad,dtype=[dtype])

#hessalt = tf.stack(tf.gradients(tf.unstack(tf.reshape(tf.gradients(l,rtheta)[0],[1,-1]),axis=0),rtheta),axis=0)



invhess = tf.matrix_inverse(hess)
sigmas = tf.sqrt(tf.diag_part(invhess))

opt = tf.contrib.opt.NadamOptimizer().minimize(l)


#initialize output tree
f = ROOT.TFile( 'fitresults_%i.root' % seed, 'recreate' )
tree = ROOT.TTree("fitresults", "fitresults")

tseed = array('i', [seed])
tree.Branch('seed',tseed,'seed/I')

titoy = array('i',[0])
tree.Branch('itoy',titoy,'itoy/I')

terrstatus = array('i',[0])
tree.Branch('errstatus',terrstatus,'errstatus/I')

tedmval = array('f',[0])
tree.Branch('edmval',tedmval,'edmval/F')

tsigvals = []
tsigerrs = []
for sig in signals:
  tsigval = array('f', [0.])
  tsigerr = array('f', [0.])
  tsigvals.append(tsigval)
  tsigerrs.append(tsigerr)
  tree.Branch(sig, tsigval, '%s/F' % sig)
  tree.Branch('%s_err' % sig, tsigerr, '%s_err/F' % sig)

tthetavals = []
ttheta0vals = []
tthetaerrs = []
for syst in DC.systs:
  systname = syst[0]
  tthetaval = array('f', [0.])
  ttheta0val = array('f', [0.])
  tthetaerr = array('f', [0.])
  tthetavals.append(tthetaval)
  ttheta0vals.append(ttheta0val)
  tthetaerrs.append(tthetaerr)
  tree.Branch(systname, tthetaval, '%s/F' % systname)
  tree.Branch('%s_In' % systname, ttheta0val, '%s_In/F' % systname)
  tree.Branch('%s_err' % systname, tthetaerr, '%s_err/F' % systname)

#initialize tf session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


#print("grad")
#print(sess.run(grad))

#print("hess")
#print(sess.run(hess))

##print("debug")
##print(sess.run(gradsq))

#print("hessalt:")
#print(sess.run(hessalt))

#exit(0)


ntoys = options.toys
if ntoys <= 0:
  ntoys = 1

#prefit to data if needed
if options.toys>0 and options.toysFrequentist and not options.bypassFrequentistFit:
      tf.contrib.opt.ScipyOptimizerInterface(l, options={'disp': True, 'gtol' : 0., 'edmtol': options.tolerance}, method=minimize_bfgs_custom).minimize(sess)
      rthetav = sess.run(thetav)

def printfunc(l):
  print(l)

def printstep(x,y):
  print([x,y])

for itoy in range(ntoys):
  titoy[0] = itoy
  
  sess.run(rtheta.assign(rthetav))
  
  #apply PCA from asimov toy
  sess.run(nobs.assign(nexp))
  
  xval, gradval, hessval = sess.run([rtheta,grad,hess])
    
  invhess = np.linalg.inv(hessval)
  eigvals, eigvects = np.linalg.eigh(invhess)

  rotval = np.transpose(eigvects/np.sqrt(eigvals))
  
  sess.run(rtheta.assign(np.reshape(np.matmul(rotval,np.reshape(rthetav,[-1,1])),[-1])))
  sess.run(rotation.assign(np.linalg.inv(rotval)))  
  
  
  xval, gradval, hessval = sess.run([rtheta,grad,hess])
  print("new hessval:")
  print(hessval)  
  
  if options.toys < 0:
    print("Running fit to asimov toy")
    sess.run(nobs.assign(nexp))
    #sess.run(rtheta.assign(1.1*rthetav))
  elif options.toys == 0:
    print("Running fit to observed data")
    sess.run(nobs.assign(data_obs))
  else:
    print("Running toy %i" % itoy)
    if options.bootstrapData:
      #randomize from observed data
      sess.run(nobs.assign(tf.random_poisson(nobs,shape=[],dtype=dtype)))
    else:
      #randomize from expectation
      sess.run(nobs.assign(tf.random_poisson(nexp,shape=[],dtype=dtype)))  
  
    if options.toysFrequentist:
      #randomize nuisance constraint minima
      sess.run(theta0.assign(theta + tf.random_normal(shape=thetav.shape,dtype=dtype)))
      thetatmp = sess.run(theta0)
      rthetatmp = np.concatenate((rthetav[:npoi],thetatmp),axis=0)
      sess.run(rtheta.assign(rthetatmp))
    else:
      #randomize actual values (TODO)
      pass
  
  xval, gradval, hessval = sess.run([rtheta,grad,hess])
  ##try:
    ##chol = np.linalg.cholesky(hessval)
    ##isconvex = True
  ##except np.linalg.LinAlgError:
    ##isconvex = False
    
  #invhess = np.linalg.inv(hessval)
  #eigvals, eigvects = np.linalg.eigh(invhess)
  
  #print("eigvals")
  #print(eigvals)
  #print("eigvects")
  #print(eigvects)
  
  
  ##for vect,val in zip(eigvects,eigvals):
    ##vect *= 1./val/val
  ##eigvects /= np.sqrt(eigvals)
  #rotval = np.transpose(eigvects/np.sqrt(eigvals))

  ##eigvects = eigvects/np.sqrt(np.reshape(eigvals))

  ##eigvects = np.matmul(np.diagflat(1./np.sqrt(eigvals)),eigvects)

  
  #sess.run(rtheta.assign(np.reshape(np.matmul(rotval,np.reshape(rthetav,[-1,1])),[-1])))
  #sess.run(rotation.assign(np.linalg.inv(rotval)))
           
           

  
  
  print(np.linalg.eigvalsh(hessval))
  #isconvex = np.all(np.greater_equal(np.linalg.eigvalsh(hessval),0.))
  isconvex = np.all(np.greater(np.linalg.eigvalsh(hessval),-1e-6))
  print("isconvex: %r" % isconvex)
  print("condition = %e" % np.linalg.cond(hessval))
  
  try:
    invhess = np.linalg.inv(hessval)
    sigmasv = np.sqrt(np.diag(invhess))
    edmval = 0.5*np.matmul(np.matmul(np.transpose(gradval),invhess),gradval)
    errstatus = 0
    if np.any(np.isnan(sigmasv)):
      errstatus = 1
  except np.linalg.LinAlgError:
    sigmasv = -99.*np.ones_like(xval)
    edmval = -99.
    errstatus = 2    
    
  print("full edmval = %e" % edmval)  
  
  #if not isconvex:
    
    #for i in range(2000):
      #lo, _ = sess.run([l, opt])
      ##_, lo = sess.run( [rtheta.assign(rtheta - tf.reshape( tf.matrix_triangular_solve(hess,tf.reshape(grad,[-1,1])),[-1]) ), l] )
      #print(lo)
  
    #xval, gradval, hessval = sess.run([rtheta,grad,hess])
    
    #print(np.linalg.eigvalsh(hessval))
    #isconvex = np.all(np.greater(np.linalg.eigvalsh(hessval),-1e-6))
    #print("isconvex: %r" % isconvex)
    
    ##try:
      ##chol = np.linalg.cholesky(hessval)
      ##isconvex = True
    ##except np.linalg.LinAlgError:
      ##isconvex = False
  ##print(sess.run(nexp))
  ##print(sess.run(nobs))
  
  
    #print("isconvex: %r" % isconvex)
  
  #minimizer = tf.contrib.opt.ScipyOptimizerInterface(l, options={'disp': True, 'gtol' : 0., 'edmtol': options.tolerance}, method=minimize_bfgs_custom)
  #minimizer = tf.contrib.opt.ScipyOptimizerInterface(l, options={'disp': True, 'gtol' : 0., 'edmtol': options.tolerance}, method=minimize_bfgs_custom,fetches = [l], loss_callback=printfunc)

  #minimizer = tf.contrib.opt.ScipyOptimizerInterface(l, options={'disp': True, 'hess' : scipy.optimize.SR1()}, method=_minimize_trustregion_constr)
  #minimizer = ScipyHessOptimizerInterface(l, options={'disp': True, 'maxiter' : 100000}, method='trust-constr')
  
  
  edmtolfinal = 1e-3
  edmtol = 0.01*edmtolfinal
  for ifit in range(1):
    minimizer = ScipyHessOptimizerInterface(l, options={'disp': True, 'maxiter' : 10000, 'gtol' : 0., 'xtol' : 0., 'barrier_tol' : 0., 'edmtol' : 1e-5}, method=_minimize_trustregion_constr_custom)
    
    #minimizer = ScipyHessOptimizerInterface(l, options={'disp': True, 'gtol' : 0., 'edmtol': options.tolerance}, method=minimize_bfgs_custom)
    
    
    ret = minimizer.minimize(sess,fetches=[l],loss_callback=printfunc)
    #ret = minimizer.minimize(sess,fetches=[l])


    xval, gradval, hessval = sess.run([rtheta,grad,hess])
  
    print(np.linalg.eigvalsh(hessval))
    isconvex = np.all(np.greater(np.linalg.eigvalsh(hessval),-1e-6))
    isconvexstrict = np.all(np.greater_equal(np.linalg.eigvalsh(hessval),0.))
    print("isconvex: %r" % isconvex)
    print("isconvexstrict: %r" % isconvexstrict)
    
    try:
      invhess = np.linalg.inv(hessval)
      sigmasv = np.sqrt(np.diag(invhess))
      edmval = 0.5*np.matmul(np.matmul(np.transpose(gradval),invhess),gradval)
      errstatus = 0
      if np.any(np.isnan(sigmasv)):
        errstatus = 1
    except np.linalg.LinAlgError:
      sigmasv = -99.*np.ones_like(xval)
      edmval = -99.
      errstatus = 2    
      
    print("full edmval = %e" % edmval)
    
    
    
    if isconvexstrict and edmval > 0. and edmval<edmtolfinal:
      break
    
    #if edmtol>0.1*edmtolfinal:
      #edmtol/=10
    
    #invhess = np.linalg.inv(hessval)
    #edmval = 0.5*np.matmul(np.matmul(np.transpose(gradval),invhess),gradval)

    
    #edmtol /= 10.
  
  #ret = minimizer.minimize(sess)
  #ret = minimizer.minimize(sess,fetches=[l],loss_callback=printfunc)
  #ret = minimizer.minimize(sess,fetches=[l],loss_callback=printfunc,step_callback=printstep)

  #get fit results
  #xval, gradval, hessval = sess.run([rtheta,grad,hess])

  #try:
    #chol = np.linalg.cholesky(hessval)
    #isconvex = True
  #except np.linalg.LinAlgError:
    #isconvex = False

  #print("isconvex: %r" % isconvex)

  
  #compute uncertainties and diagnostics  
  try:
    invhess = np.linalg.inv(hessval)
    sigmasv = np.sqrt(np.diag(invhess))
    edmval = 0.5*np.matmul(np.matmul(np.transpose(gradval),invhess),gradval)
    errstatus = 0
    if np.any(np.isnan(sigmasv)):
      errstatus = 1
  except np.linalg.LinAlgError:
    sigmasv = -99.*np.ones_like(xval)
    edmval = -99.
    errstatus = 2
    
  print("errstatus = %i, edmval = %e" % (errstatus,edmval))
  
  terrstatus[0] = errstatus
  tedmval[0] = edmval
  
  #get fit values
  sigvals = xval[:npoi]
  thetavals = xval[npoi:]

  rsigmasv = sigmasv[:npoi]
  thetasigmasv = sigmasv[npoi:]
  
  theta0vals = sess.run(theta0)

  for sig,sigval,sigma,tsigval,tsigerr in zip(signals,sigvals,rsigmasv,tsigvals,tsigerrs):
    tsigval[0] = sigval
    tsigerr[0] = sigma
    if itoy==0:
      print('%s = %f +- %f' % (sig,sigval,sigma))

  for syst,thetaval,theta0val,sigma,tthetaval,ttheta0val,tthetaerr in zip(DC.systs,thetavals,theta0vals,thetasigmasv,tthetavals,ttheta0vals,tthetaerrs):
    tthetaval[0] = thetaval
    ttheta0val[0] = theta0val
    tthetaerr[0] = sigma
    if itoy==0:
      print('%s = %f +- %f' % (syst[0], thetaval, sigma))
    
  tree.Fill()

f.Write()
f.Close()