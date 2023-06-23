
rm(list=ls())
setwd('/Users/wenrurumon/Documents/postdoc/mtard/data/hexin0926')
library(data.table)
library(segmented)
library(dplyr)
library(igraph)
library(qgraph)
library(msigdbr)
library(fgsea)
library(ggplot2)
library(impute)

pca <- function(A){
  A.svd <- svd(A)
  d <- A.svd$d
  d <- d[d > 1e-10]
  r <- length(d)
  prop <- d^2; prop <- cumsum(prop/sum(prop))
  d <- diag(d,length(d),length(d))
  u <- A.svd$u[,1:r,drop=F]
  v <- A.svd$v[,1:r,drop=F]
  x <- u%*%sqrt(d)
  y <- sqrt(d)%*%t(v)
  z <- x %*% y
  rlt <- list(X=x,Y=y,Z=x%*%y,prop=prop)
  return(rlt)
}

#Load Data

system.time(raw <- readRDS('data.rds'))
map <- read.csv('map0926.csv')[,-1]  %>%
  mutate(pheno=paste(pheno),omic=paste(omic),subomic=paste(subomic),
         class1=paste(class1),class2=paste(class2),group=paste(group)) %>%
  mutate(class2=ifelse(class2=='TCM ','TCM',class2)) %>%
  mutate(class1=ifelse(class1=='DXA ','DXA',class1)) %>%
  mutate(class1=ifelse(grepl('Skin',class1,ignore.case=T),'Skin',class1)) %>%
  mutate(group=ifelse(class1=='Energy metabolism','X',group)) %>%
  mutate(class2=ifelse(subomic=='分子代谢-BileAcidsFecal','Bile acids fecal',class2)) %>%
  mutate(item=substr(pheno,regexpr(':',pheno)+1,nchar(pheno))) %>%
  mutate(group=ifelse((class1=='Sleep')&(grepl('score',class2)),'Y',group)) %>%
  mutate(class2=ifelse((class1=='Sleep')&(grepl('score',class2)),'Sleep questionnaire',class2)) %>%
  mutate(class2=ifelse((class1=='Sleep')&(group=='E'),'Sleep',class2)) %>%
  mutate(class1=ifelse((group=='E')&(class2=='Sleep'),'Health questionnaire',class1)) %>%
  mutate(group=ifelse(class1%in%c('Depressive','Anxiety'),'Y',group)) %>%
  mutate(class1=ifelse(class1%in%c('Depressive','Anxiety'),'Mood',class1)) %>%
  mutate(class2=ifelse(subomic=='基本信息','Age&Gender',class2)) %>%
  mutate(class2=ifelse(class2=='3D body scan ','3D body scan',class2)) %>%
  mutate(group=ifelse(class2=='TCM','rm',group)) %>%
  mutate(group=ifelse(class2=='Ultrasonic detector','rm',group)) %>%
  mutate(group=ifelse(class1=='Biochemistry','X',group)) %>%
  mutate(class2=ifelse(class1%in%c('Anxiety','Depressive'),'Mood',class2)) %>%
  mutate(class1=ifelse(class1%in%c('Anxiety','Depressive'),'Mood',class1)) %>%
  mutate(group=ifelse(class2=='ExRNA','rm',group)) %>%
  mutate(group=ifelse(class2=='Psychology assessment','rm',group)) %>%
  mutate(class2=ifelse(class2=='Ophthalmic optical coherence tomography scanner','EyeScanner',class2)) %>%
  mutate(class2=ifelse(class2=='Otoacoustic emission tester','Hearing',class2)) %>%
  mutate(class1=ifelse((group%in%c('X','E'))|(class1%in%c('Skin','Mood')),class1,class2)) %>%
  mutate(pheno=paste(class1,class2,item,sep=':')) 
map <- map[-which(rownames(raw)%in%c('志愿者招募-常规体检:身高（cm）','志愿者招募-常规体检:体重（kg）')),]
raw <- raw[-which(rownames(raw)%in%c('志愿者招募-常规体检:身高（cm）','志愿者招募-常规体检:体重（kg）')),]
rownames(raw) <- map$pheno

gender <- apply(t(raw[1:2,] %>% as.matrix),2,as.numeric)
age <- raw[3,] %>% as.numeric()
sel <- (map$group=='Y')|(map$class1%in%c('Proteome','Metabolome','RNA'))
raw <- raw[sel,]
map <- map %>% filter(pheno%in%rownames(raw))
system.time(raw <- apply(raw,2,as.numeric))
rownames(raw) <- map$pheno

map$obs <- rowMeans(!is.na(raw))
map %>% 
  filter(group=='X') %>%
  group_by(omic,subomic,class1,class2) %>% 
  summarise(n=n(),obs=mean(obs)) %>% 
  arrange(desc(n)) 

#Replace NA with 0 for tested samples

for(i in unique(filter(map,group=='X')$class2)){
  sel <- which(map$class2==i)
  x <- raw[sel,which(colMeans(is.na(raw[sel,]))<1)]
  x[x<0] <- 0
  x <- ifelse(is.na(x),0,x)
  x[x==0] <- min(x[x>0])/2
  raw[sel,which(colMeans(is.na(raw[sel,]))<1)] <- log(x)
}
raw <- t(raw)
map$obs <- colMeans(is.na(raw))
raw <- raw[,map$obs<0.8]
map <- map %>% filter(obs<0.8)
raw.sd <- apply(raw,2,sd,na.rm=T)
raw <- raw[,raw.sd>0,drop=F]
map <- map %>% filter(pheno%in%colnames(raw))

# Age Related Pheno

raw.corp <- apply(raw,2,function(y){
  r <- cor.test(y,age,use='pairwise')
  c(r$estimate,pvalue=r$p.value)
}) %>% t

map <- map %>% 
  cbind(raw.corp) %>%
  mutate(pfdr=p.adjust(pvalue,method='fdr'))

# map %>%
#   filter(group=='X',class1=='RNA') %>%
#   ggplot() + 
#   geom_point(aes(x=pheno,y=-log(pfdr,base=10),colour=subomic),size=.05) +
#   geom_hline(yintercept=-log(0.05,base=10),linetype=2) +
#   theme(legend.position='top',axis.text.x=element_text(size=0)) +
#   labs(x='Phenotypes',y='-LogPvalue')

#Impute data

Y <- raw[,filter(map,group=='Y',pfdr<0.05)$pheno]
# X <- raw[,filter(map,class1%in%c('RNA'))$pheno]
X <- raw[,filter(map,class1%in%c('Proteome','Metabolome','RNA'),pfdr<0.05)$pheno]
YX <- cbind(Y,X)
Z <- cbind(age,gender)
Z <- Z[rowMeans(is.na(YX))<=0.2,]
YX <- YX[rowMeans(is.na(YX))<=0.2,]

library(impute)
YX2 <- impute.knn(cbind(Z,YX))[[1]][,-1:-3] %>% scale

#OPU

opu <- lapply(unique(filter(map,group=='Y',pheno%in%colnames(YX2))$class2),function(i){
  print(i)
  mapi <- map %>% filter(class2==i,pheno%in%colnames(YX2))
  yi <- YX2[,mapi$pheno,drop=F] 
  g <- (abs(cor(yi))>0.95) %>%
    graph_from_adjacency_matrix(mode='undirected')
  g <- components(g)$membership
  yi <- sapply(unique(g),function(j){
    pca(YX2[,names(which(g==j)),drop=F])$X[,1]
  })
  colnames(yi) <- paste0(i,'_',unique(g))
  mapi <- mapi %>% mutate(opu=paste0(class2,'_',g))
  list(y=yi,map=mapi)
})

xpu <- lapply(unique(filter(map,group=='X',pheno%in%colnames(YX2))$class2),function(i){
  print(i)
  mapi <- map %>% filter(class2==i,pheno%in%colnames(YX2))
  yi <- YX2[,mapi$pheno,drop=F] 
  g <- (abs(cor(yi))>0.95) %>%
    graph_from_adjacency_matrix(mode='undirected')
  g <- components(g)$membership
  yi <- sapply(unique(g),function(j){
    pca(YX2[,names(which(g==j)),drop=F])$X[,1]
  })
  colnames(yi) <- paste0(i,'_',unique(g))
  mapi <- mapi %>% mutate(opu=paste0(class2,'_',g))
  list(y=yi,map=mapi)
})

#Merge

Y <- do.call(cbind,lapply(opu,function(x){x$y}))
X <- do.call(cbind,lapply(xpu,function(x){x$y}))
colnames(Z) <- c('age','male','female')
mapu <- rbind(
  do.call(rbind,lapply(opu,function(x){x$map})) %>%
    select(pheno,omic,subomic,class1,class2,group,item,pfdr,opu),
  do.call(rbind,lapply(xpu,function(x){x$map})) %>%
    select(pheno,omic,subomic,class1,class2,group,item,pfdr,opu)
)
Y2 <- apply(Y,2,function(x){x * sign(cor(x,Z[,1]))})

#Function and Organ

mapu2 <- mapu %>%
  filter(group=='Y') %>%
  group_by(group,class1,class2) %>%
  summarise(n=n()) %>%
  as.data.frame

mapu <- mapu %>%
  mutate(group=ifelse(class2%in%mapu2$class2[c(8,11,12,13,15,17,20:24,34:36,38)],'F',group))

#Output

Ystruct <- Y2[,colnames(Y2)%in%(mapu %>% filter(group=='Y'))$opu]
Yfunc <- Y2[,colnames(Y2)%in%(mapu %>% filter(group=='F'))$opu]
X.age <- X[,p.adjust(apply(X,2,function(x){cor.test(x,Z[,1])$p.value}),method='fdr')<0.05,drop=F]
X.age2 <- X.age[,!grepl('Whole blood',colnames(X.age))]
X.prot <- X[,grep('^Proteome',colnames(X)),drop=F]
X.rna <- X[,grep('^Whole blood',colnames(X)),drop=F]

setwd('/Users/wenrurumon/Documents/postdoc/mtard/data/data0211')
write.csv(Y2,'Y.csv')
write.csv(Ystruct,'Ystruct.csv')
write.csv(Yfunc,'Yfunc.csv')
write.csv(X.age,'Xage.csv')
write.csv(X.age2,'Xage2.csv')
write.csv(X.prot,'Xprot.csv')
write.csv(X.rna,'Xrna.csv')
write.csv(Z,'Z.csv')
save(X.age,X.prot,X.rna,Y2,Ystruct,Yfunc,Z,mapu,file='data.rda')

#COV for Glasso

sel <- sapply(0:15,function(i){sum((Z[,1]>=i*2+20)&(Z[,1]<=i*2+30))})
test <- sapply(sel,function(n){
  sapply(1:10,function(s){
    set.seed(s)
    yi <- Y[sample(1:nrow(Y))[1:n],]
    covi <- cor(yi)
    quantile(abs(covi),0.9,na.rm=T)
  })
})
# write.csv(test,'covquantile90.csv')

melt(test) %>%
  ggplot() + 
  geom_point(aes(x=Var2,y=value)) +
  geom_smooth(aes(x=Var2,y=value))
