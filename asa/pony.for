c Program CATPAC
c pony orf a while
c now Catpac4 6/19/19 JW
c Joe Woelfel
c Scott Danielsen
c
c *** PC Version 1.00 *****

c        error routine for unique word limit 60000


c       write(tline(22:26),'(i5)')num

c    must check for ascii range of alpha numeric chars put in tin
c    also working on specifing help.


c   SD 3-4-91 to 3-12-91 made changes to allow compatibility in
c   IBMland and trim down size.  Compatible with Microsoft Fortran 5.0.
c   Major changes include modularcall structure, and new kounting logic
c   to cut down on matrix storage

c   SD 3-13-91 added window routines.  You can have any size window with
c   any size slide or still case delimit with -1.

c   SD- Trans and f(i,j) laid to rest this day 3-15-91 (check out ccount).
c   f(i,j) premultiplied by its transpose = sum of each vector f(j)
c   premulitplied by it's transpose.  Count of words in each episode is read
c   in as vector f, is pre-multiplied by it's transpose and summed to win.
c   You may now have a whole lot of cases.

c   SD- 3-18-91 added net and attendant routines.  It works like this
c   catalac will learn the words and spit out a .win and .lbl file.

c   SD - 3-26-91 increased unique word capacity from 100 to 300.  This
c   makes more sense if you are analyzing books, or large text db's.
c   Size of unique word limiter (nlim) is user selected: 100 <= nlim <= 300.

c   SD - 11-4-91 standardized interface (====), added default prompting [ ],
c   added threshold change for 0 - 1 logistic(.5)  cleaned up help files (cp_)

c   SD - 7-17-92 added bigeig to produce crd files for plotting

c   SD - 7-19-92 added freqprint to print out frequency counts of all words.

c   SD - 7-21-92 added coprint to print out cooccurence matrix.

c   SD - 8-7-92 fixed misspelling of cooccurence and removed prompt for
c               cooccurence print.
c	jw - 10/22/15 modified to run on OS-X
c   jw made into pony January 2015 Mainly includes log transform in 
c		subroutine squash, which spreads out catpac spaces nicely...

c        use msflib

      parameter (msize=5000)

      common /in/win,ncon
      common /stuff/nlines,nuniqe,ntepi,nepi,nwrds,nwords
c
      logical isnet,isfreq,isco,codef,freqdef,netdef
      real*4 win(msize,msize)
      character*40 message,tit*80,clamp*3,line*1000
        character*1 key

c      integer*2 getche

      data ncycle/1/,clamp/'Yes'/,nepi/0/
c
c     The main controlling program.

 10   continue

      isnet=.false.
      isfreq=.false.
      isco=.false.

      call reset(nlines,ntepi,nwrds,nuniqe,nwords)

      call menu(*44,*22,*99,*11,*55,*99,*99)

  11  call freqparms(tit,freqdef,*10)
      ngo=1
      isfreq=.true.
      go to 88

 22   continue
c      call cooparms(tit,nlim,isco,codef,*10)
c      ngo=2
c      isnet=.false.
c      go to 88
c
  44  call netparms(tit,nlim,isnet,ncycle,clamp,netdef,*10)
      ngo=3
      go to 88

c **********************************************
c    Call analysis routines
c **********************************************

  55  freqdef=.true.
      codef=.true.
      netdef=.true.
      go to 10
      
88    continue     
c        88  call timer(istart)

c      call catwrt('CATPAC_PC','v2.00',3,2)
c      if(.not.isfreq)call exclude
      call exclude
c      call progress(' ',100)
      call status('Scanning, tome breath')
    3 call read(line,nlines,ntepi,*4)
      call parse(line,nuniqe,nwrds,*3,*4)
      
   4   continue
c   4   call progress('                 ',100)
      if(isfreq)then
      call prints(nepi,nuniqe,nlines,nwrds,tit,isnet,ncycle,clamp,
     +isfreq)
      go to 2
      endif

      call sort(nuniqe,nwords,nwrds,ntepi,nlim)
      nepi=0
      rewind(14)
      if(isnet)call net(nepi,nuniqe,ncycle,ntepi,isnet,*5)
      call ccount(nepi,nuniqe,isnet,ntepi,*5,nold,.false.)
 5    if(isco)call coprint(win,nuniqe,nwrds,nlines,tit)

      if(isnet)call crdmake

      call prints(nepi,nuniqe,nlines,nwords,tit,isnet,ncycle,clamp,
     +isfreq)

      call hiclus(nuniqe)

   2  continue
      close(2)
      close(3)
      close(9)
      close(14,status='delete')
      close(16)
      close(17)
      close(20)

      call reset(nlines,ntepi,nwrds,nuniqe,nwords)
c      call timer(iend)
      rfin=(iend-istart)/60.

      out=380
      if(isnet)out=420
      message='All done. '
c      call moveto(10,out,xy)
      write(message(26:32),'(f7.2)')
c      message(33:40)=' minutes.'
      print*,message
c      print*,message
c     call moveto(10,out+20,xy)
      print*,'Sh bob, sh bop, sh bang.'
c      call moveto(10,out+45,xy)
      print*,'Press any key to continue.'
      read(*,*)
c end call analysis routines

c *************************************
      go to (11,22,44)ngo

  99  continue

       end

      subroutine menu(*,*,*,*,*,*,*)

c     displays and returns the main menu selections

      character*45 menu_options(7)
     +/'1 - Network Analysis',
     + '2 - Cluster Analysis',
     + '3 - Help',
     + '4 - Frequencies',
     + '5 - Reset to defaults',
     + ' ',
     + '7 - Exit'/

      character*80 hfile

      hfile='\galileo\help\cp_help.hlp'

   1     write(*,100)menu_options
 100  format(1x,a45)
         read(*,101,err=1)nopt

 101   format(i1)
      if(nopt.ge.1.and.nopt.le.7)then
          return nopt
        else
          go to 1
        endif
      end


c      subroutine decide(i,nopt,*)
c        use msflib
c        character*1 key
c      decides if key pressed is a function key
c
c      nopt=0
c        if(i.eq.0)then
c        KEY=getcharqq()
c        i=ichar(key)
c
c      if(i.ge.59.and.i.le.68)nopt=i-68+10
c
c      return 1
c      endif
c   10 continue
c      return
c      end

      subroutine read(line,nlines,ntepi,*)
      common /window/iswin,winsize,winwrds,nslide
      logical iswin
      integer winsize,winwrds
      character*1000 line
c
    3 read(2,555,end=55)line
      if (line .eq.' ')go to 3
      nl=nl+1
      if(line.eq.'-1'.and. iswin)go to 3
      if (line.eq.'-1')then
        if(nl.eq. 1)go to 3
        write(14,556)line
        ntepi=ntepi+1
        nl=0
        go to 3
      endif
      nlines=nlines+1
c      write(*,1112)nlines   ! ** debug **
c 1112 format('+ Reading dataa: line ',i7,'               ')  ! ** debug **
      return

cc   55 call progress('Scanning',100)
   55 return 1
  555 format(a1000)
  556 format(a2)
      end
c
      subroutine parse(ans,nuniqe,nwords,*,*)
c
c     This will parse a user response
c     change letters to upperc, delete unecessary words
c     and  keep an initial tally of words and frequency.
c
      common /locate/isinc,iwords,niwords,iself,nmethod
      character*15 word,idword,iwords(200)
      logical isinc,iself
      character*1000 ans
      data nend/1000/

c
      nx=1
      num=0

    1 do 10 i=nx,nend
         num=num+1
         ntest=ichar(ans(i:i))

        if(ans(i:i).eq.'''')go to 10 !FOR STOCKS

        if(ans(i:i) .eq. '_')go to 10 ! for stocks

c       sd-10-26-92 + or - for first step to easy cloud seeding

C        if(ans(i:i).eq.'+'.or.ans(i:i).eq.'*')go to 10  ! FOR STOCKS
c        if(ntest.ge.48.and.ntest.le.57)go to 10    !numeric
c
        if(ntest.le.64.and.ntest.ge.91)go to 9
        if(ntest.ge.97.and.ntest.le.122) ans(i:i)=char(ntest-32)

        if(ichar(ans(i:i)).ge.65.and.ichar(ans(i:i)).le.90)go to 10

    5      if(num .le. 1)go to 9

           word=ans(nx:i-1)

           if(word(1:1).eq.'+'.or.word(1:1).eq.'*')idword=word

c          check if self ref, if self-locate change word to idword.

           if(iself.and.nmethod.eq.2)then
           do 13 j = 1,niwords
   13      if(word.eq.iwords(j))word=idword
           endif

c          end check if self ref

           call delete(word,*9)
           nwords=nwords+1
    8      call icount(word,nuniqe,*55)
           write(14,100)word
    9      if (ans(i+1:nend).eq.' ')return 1
           nx=i+1
           num=0
         go to 1
   10 continue
      return 1
  100 format(a15)
   55 return 2
      end
c
      subroutine delete(word,*)
      common /del/nwrds,Bwords

c
c     deletes uneeded words.
c


      character*15 bwords(1000),word
    1 format(a15)

      do 5 k=1,nwrds
      if(word.eq.bwords(k)) return 1
    5 continue
      return
      end
c
      subroutine icount(word,nuniqe,*)
      common /freq/kwords
      common /alpha/words
      dimension kwords(60000)
      character*15 words(60000),word
c

c     will keep and initial tally of words and kwords.
c
      do 6 J=1,nuniqe
        if(word.eq.words(j))go to 7
    6 continue
c
      nuniqe=nuniqe+1
      if(nuniqe.gt.60000)then
         write(*,1312)
 1312 format(/'+ Unique word limit of 60000 has been reached.')
         return 1
      endif
c
      words(nuniqe)=word
      kwords(nuniqe)=kwords(nuniqe)+1
  100  format(a15)
      return
    7 kwords(j)=kwords(j)+1
      return
      end
c
      subroutine sort(nuniqe,nwords,nwrds,ntepi,nlim)
      parameter (msize=5000)
      common /freq/kwords
      common /alpha/words
      common /window/iswin,winsize,winwrds,nslide
      common /locate/isinc,iwords,niwords,iself,nmethod

        real tepi
      logical iswin,isinc,iself
      integer winsize,winwrds
      character*15 words(60000),ait,iwords(200)
      dimension kwords(60000)
c
c     This will sort kwords in descending frequency and strip to nlim


c     set generated vars to null for multiple runs

      rwords=0
c
c     first find total episodes for output purposes
c

      if(iswin)then
      tepi=(nwrds+nslide-winsize)/nslide
      ntepi=tepi-amod(tepi,1.)
      endif
c
      nwords=0
      npct=0
102   flip = 0
c      num=num+1
      do 103 i=1,nuniqe-1
        if(kwords(i).ge.kwords(i+1))go to 103
        flip=1
        it=kwords(i)
        kwords(i)=kwords(i+1)
        kwords(i+1)=it
        ait=words(i)
        words(i)=words(i+1)
        words(i+1)=ait
  103 continue
      if(npct.ge.95)npct=0
c      call progress('Descending, Dante',npct)
      npct=npct+2
      if(flip.ne.0)go to 102
c
c     now strip em
c
c      call progress(' ',100)

      ntotwrds=nuniqe
      nuniqe=0
      do 10 i = 1,nlim
      new=0
      j=i
      do while(kwords(j).eq.kwords(i))
       j=j+1
       new=new+1
       if(nuniqe+new.gt.nlim)go to 11
  9   enddo
      nuniqe=nuniqe+1
c      tfreq(i) = kwords(i)
c      twords(i)= words(i)
c      rwords=rwords + tfreq(i)
      rwords=rwords + kwords(i)
 10   continue


c      add id words to word list sd -10-27-92  cloud seeding
c     we may want to make this more general with an 'include'
c     file that supercedes stripper.  Please note that (and this should
c     be maintained if we go to an include file) words are added AFTER
c     stripping.  This way any subsest of nlim words can be associated
c     with the id's or whatever words are to be 'included'.
c

  11  do 17 i = nuniqe+1,ntotwrds

c sd 4-27-93 to prevent including + or * words if not zelf anal.

c      if(words(i)(1:1).eq.'+' .or. words(i)(1:1).eq.'*')go to 19
      if(iself.and.(words(i)(1:1).eq.'+' .or. words(i)(1:1).eq.'*'))
     +go to 19

c     check for include words if needed (.not. iself)

      if(isinc)then
      do 18 j = 1,niwords
  18  if(words(i).eq.iwords(j))go to 19    !include words check
      endif
      go to 17

  19    if(i.le.nuniqe)go to 17
        nuniqe=nuniqe+1
        kwords(nuniqe)=kwords(i)
        words(nuniqe)=words(i)
c        tfreq(nuniqe)=kwords(i)
c        twords(nuniqe)=words(i)
        rwords=rwords+kwords(nuniqe)

   17 continue

c     end add id words to word list

      nwords=rwords
      return
      end
c
      subroutine ccount(nepi,nuniqe,isnet,ntepi,*,nold,iself)
      parameter (msize=5000)
      common /vector/ext
      common /freq/kwords
      common /alpha/words
      common /in/win,ncon
      logical iswin,isnet,iself
      common /window/iswin,winsize,winwrds,nslide
      common /casecnt/casecnt

      real*4 win(msize,msize),ext(msize),casecnt(msize)
      integer winsize,winwrds,kwords(60000)
      character*15 word,words(60000)
c
c                 print67
c67    format('+ in ccount')


  6   if(iswin)go to 3
  1   read(14,100,end=55)word
 100  format(a15)
      if(word.eq.'-1')then
        nepi=nepi+1
        ntepi=ntepi-1
        go to 2
      endif
c
      do 15 i=1,nuniqe
      if(word.eq.words(i))then
        ext(i)=ext(i)+1
        if(ext(i).eq.1)casecnt(i)=casecnt(i)+1
      endif
   15 continue
      go to 1
c
   3  do 11 i = 1,winsize
        read(14,100,end=55)word

c       sd 10-27-92  cloud seeding

c       check to see if this is an id word. if it is, then turn off old
c       neuron, set nold to present id.  it will turn on itself and
c       stay on w/clamping,  we can force it as well (see cycle).

        do 115 j=1,nuniqe
         if(word.eq.words(j))then

c sd 4-27-93 to prevent including + or * words if not zelf anal.
c          if(word(1:1).eq.'+'.or.word(1:1).eq.'*')then
c
          if(iself.and.(word(1:1).eq.'+'.or.word(1:1).eq.'*'))then
            if(nold.ne.0)ext(nold) = 0.0
            nold=i
          endif

c     end check if id word
           ext(j)=ext(j)+1
           if(ext(j).eq.1)casecnt(j)=casecnt(j)+1
         endif
  115   continue

c     end check if id word


  11  continue
      nepi=nepi+1
      ntepi=ntepi-1
      do 12 i = 1,winsize-nslide
  12  backspace(14)
c

 2    npct=aint(real(nepi)/real(nepi+ntepi)*100.)
c      call progress('Self Organizing',npct)   ! ** debug **
c      if(mod(npct,2).eq.0)call progress('Self Organizing',npct)

      if(isnet)return   ! go back if nework other wise do co-occ below

      do 13 i=1,nuniqe
      do 17 j=1,nuniqe
      win(i,j)=win(i,j)+ext(i)*ext(j)
   17 continue
   13 continue
      do 16 i = 1,nuniqe
  16  ext(i)=0
      go to 6
  55  return 1
      end
c
      subroutine hiclus(n)
      parameter (msize=5000)
c
c
c     sd 3-20-91  e(100,200) = win(100,100)
c

C      CURRENTLY, THE MAXIMUM AND MINIMUM METHODS ARE OUTPUTED
C       OTHER METHODS (SUCH AS AVERAGEING) MAY BE ADDED

C       I HAVE TRIED TO INDICATE WHERE YOU WOULD ADD THESE

      common /in/win,ncon
      common /alpha/words3
c
      LOGICAL W(msize)
      character*15 words3(60000)
      CHARACTER*1 WORD(15,msize)
      real*4 win(msize,msize)  ! d(200,100)
      DIMENSION XX(msize),LK(msize),LKI(msize),LKJ(msize)
      DIMENSION P(600)
      DATA XXXX/1H^/, BLANK/1H /, PERIOD/1H./
      DATA is/-1/
c
       open(unit=71,file='labs.txt')

       call prntcon(15,3)      !start condensed printing.
c
C        READ IN   N = NUMBER OF VARIABLES
C                  IS = +1 IF MATRIX IS MATRIX OF DISTANCES
C                       -1 IF MATRIX IS MATRIX OF PROXIMITIES
c
c 10   READ(9,11)N,IS
C        IF N = NEGATIVE OR 0,  STOP

      IF (N .LE. 0) STOP
      S = IS
       N1=N-1
      N2=N1-1
      XN=N
c
c        READ IN TRIANGULAR MATRIX WITHOUT DIAGONAL
c        MAKE IT INTO A SYMMETRIC ARRAY WITH 0 DIAG.,  win(I,J)
c
      win(1,1)=0.0
      DO20I=2,N
      win(I,I)=0.0
      I1=I-1
      DO20J=1,I1
 20   win(J,I)=win(I,J)
c
c      catexp1.for goes here
c
c        INITIALIZE TO FIRST METHOD
c RZ 3/13/84  INITIALIZE TO SECOND METHOD

C     K=1

      K=2
C        START A METHOD

C        PRINT A TITLE

 50    GOTO(51,52),K
 51    CONTINUE !PRINT61
       write(3,61)
       GOTO98
 52    CONTINUE !PRINT62
       write(3,62)

C             SET W(I) = .FALSE.
C                 LK(I)=0
C           W(I) = .TRUE. IF LI I HAS BEEN MERGED WITH SOME OTHER PT
C           LK(I) = LIST OF PTS MERGED, FOR OUTPUT PURPOSES---SEE BELOW

 98   DO99I=1,N
      W(I) = .FALSE.
       LK(I)=0

C        COPY OFF THE ORIGINAL MATRIX INTO TEMPORARY STORAGE

C           ****   NOTE   ****

C     YOU MAY NOT WANT TO DO THIS IF PRESSED FOR SPACE

C           YOU COULD READ THE DATA DIRECTLY INTO THE D ARRAY, AND NOT
C           NEED THE SPACE FOR THE E ARRAY AT ALL

C       THEN USING TWO METHODS WOULD REQUIRE READING THE DATA TWICE

      DO99J=1,N
 99   CONTINUE
C99   D(I,J)=win(I,J)   SCOTT DANIELSEN 3-3-90, I WAS PRESSED FOR SPACE.

C        II COUNTS THE NUMBER OF PAIRS MERGED
       DO181II=1,N1

       npct=aint(real(ii)/real(n1)*100.)
       if(ii.eq.1)npct=0
c       call progress('Clustering',npct)    ! ** debug **

c      write(*,1112)ii
c 1112 format('+ Cluster analysis: iteration ',i5,'             '
c     +,'                           ')

C        FIND MIN PAIR DISTANCE


C        FIRST FIND TWO UNMERGED POINTS -- INITIALIZE X
C             X IS THE RUNNING MINIMUM DISTANCE

 100  DO102NI=2,N
      IF (W(NI)) GO TO 102
      NI1=NI-1
      DO101NJ=1,NI1
 101  IF (.NOT. W(NJ)) GO TO 104
 102  CONTINUE

C        MUST BE ONE LEFT

       STOP
 104  X=win(NI,NJ)

C        BEGIN THE REAL BUSINESS OF FINDING THE CLOSEST PAIR
C        ONLY LOOK AT UNMERGED POINTS

      DO150I=2,N
      IF (W(I)) GO TO 150
      I1 = I-1
      DO149J=1,I1
      IF (W(J)) GO TO 149

C        THIS IS THE CENTRAL COMPARE STATEMENT

      IF((win(I,J)-X)*S .GT. 0.0) GO TO 149

C        WE HAVE FOUND A CLOSER PAIR -- UPDATE X, NI, NJ

      NI = I
      NJ=J
      X=win(I,J)
 149  CONTINUE
 150  CONTINUE

C        WE HAVE NOW LOOKED AT EVERY PAIR OF POINTS --
C             STORE X IN XX, NI AND NJ IN ARRAYS LKI AND LKJ

       XX(II)=X
       LKI(II)=NI
       LKJ(II)=NJ

C        POINT NJ IS CONSIDERED 'MERGED', WHERE BY THIS IS MEANT THAT
C                  IT NO LONGER POSSESSES A SEPARATE ROW OF THE MATRIX,
C                  BUT IS ENTIRELY SUBSUMED TO NI, (WHICH IS LARGER).

      W(NJ) = .TRUE.

C        SELECT UNMERGED POINTS, L, WHICH ARE NOT NEITHER NI NOR NJ

      DO179L=1,N
      IF (W(L)) GO TO 179
      IF (L .EQ. NI) GO TO 179

C        BRANCH ACCORDING TO METHOD, AND FIND D(NI,L) AND D(L,NI)

 160  GOTO(200,300),K

C        *****MINIMUM METHOD*****

C          win(NI,L)= MIN(win(NI,L),win(NJ,L))

 200  IF((win(NI,L)-win(NJ,L))*S .LE. 0.0) GO TO 170
      win(NI,L)=win(NJ,L)
      win(L,NI)=win(L,NJ)
      GOTO170

C        *****MAXIMUM METHOD*****

C          win(NI,L)= MAX(win(NI,L),win(NJ,L))

 300  IF((win(NI,L)-win(NJ,L))*S .GE. 0.0) GO TO 170
      win(NI,L)=win(NJ,L)
      win(L,NI)=win(L,NJ)
 170  CONTINUE

C        GO GET MORE L'S

 179  CONTINUE

C        NOW GO BACK AND DO AGAIN USING NEW D ARRAY --(ITERATE)

 181   CONTINUE

C        ITERATION COMPLETE -- BEG IN PRINTOUT ROUTINE
C        WE NOW HAVE
C             1.   LKI ARRAY
C             2.   LKJ ARRAY
C                  THESE CONTANN SUCCESSIVE LINKS MERGED AT EACH
C                       ITERATI ON
C                  LKI(I) IS THE LARGER OF THE TWO
C                  LKJ(I) IS THE SMALLER OF THE TWO, THE ONE THAT IS
C                       MERGED (ELIMINATED) AT STEP I
C             3.  XX(I) = THE SIZE (VALUE) OF THE LINK MERGED AT STEP I


C        ARRANGE THE LINKS IN ORDER FOR PRINTING OUT


C        FIRST PUT THE LAST TWO POINTS CONNECTED INTO THE LK ARRAY

       LK(1)=LKJ(N1)
       LK(2)=LKI(N1)

C        THEN WORK BACKWARDS -- PICK A LINKAGE.   THE LARGER MEMBER OF
C                  THIS GROUP MUST BE ALREADY PLACED IN THE LK LIST.

C                  MOVE THE ENTRIES IN THE LK LIST OVER ONE, AND 'MAKE
C                  ROOM' FOR THE NEW ENTRY, WHICH GOES IMMEDIATELY TO
C                  THE LEFT OF THE LARGER ELEMENT.

C                  IF THESE COMMENTS ARE NOT CLEAR, IT IS IN THE COMBIN-
C                       ATORIAL NATURE OF THE SUBJECT THAT THIS BE SO,
C                       AND WORKING THROUGH A SIMPLE EXAMPLE SHOULD BE
C                       VERY EDIFYING.   IM DOING THE BEST I CAN.

       DO550II=2,N1
       JJ=N-II

C        FIND THE LKI ENTRY IN THE LK LIST

       DO510 KK=1,II
 510   IF(LKI(JJ) .EQ. LK(KK)) GO TO 515

C      IT MUST BE SOMEWHERE

       STOP

C        SHOVE THINGS OVER AND PUT THE LKJ ENTRY IN THE LK LIST

 515  DO530IK=KK,II
       KI=II+KK-IK
 530  LK(KI+1)=LK(KI)
 550  LK(KK)=LKJ(JJ)

C        OVER AND OVER IN THE DO LOOP UNTIL ALL IS DONE

C        THE LK ORDER IS THE ORDER IN WHICH THE FINAL OUTPUT WILL
C                  BE ARRANGED

C        INDEX THROUGH THE LKI AND LKJ ARRAYS, USING I

      DO585I=1,N1

C        FIND LKI(I) IN THE LK ARRAY-- THE INDEX IS JI

       DO565JI=1,N
 565   IF(LK(JI) .EQ. LKI(I)) GO TO 570

C        IT MUST BE SOMEWHERE

       STOP

C        FIND LKJ(I) IN THE LK ARRAY-- THE INDEX IS JJ

 570   DO575JJ=1,N
 575   IF(LK(JJ) .EQ. LKJ(I)) GO TO 580

C        IT MUST BE SOMEWHERE

       STOP

580   LKI(I)=JI
      LKJ(I)=JJ
585   CONTINUE

C     LKI(II)AND LKJ(II)NOW CONTAIN THE ORDINAL POSITIONS OF THE II LINK


C        THE FOLLOWING CODE MAY BE DONE MORE THAN ONCE, DEPENDING ON
C             HOW LARGE N IS -- IF N IS MORE THAN 50, THE CODE IS DONE
C             TWICE.   THE ONLY DIFFERENCE BETWEEN THE TIMES IS WHICH
C             PROTION OF THE IP AND P ARRAYS IS PRINTED OUT -- OTHERWISE
C             THE SITUATION IS IDENTICAL.

C        M1 = STARTING I VALUE
C        M2 = FINAL I VALUE
C        IP1 = STARTING P VALUE
C        IP2 = FINAL P VALUE

C        THE FIRST TIME,
C             M1 = 1, IP1=1,  M2=MIN(50,N), IP2=100 OR 2*N-1, DEPENDING
C                            AS N GREATER THAN 50, OR NOT

C        THE SECOND TIME(WHEN N GREATER THAN 50),
C             M1=51, M2=N, IP1=100, IP2=2*N-1

C        SET UP FIRST TIME
c
c      3-26-91  Scott Danielsen  changed printout routine to
c               handle up to 300 unique words (more can be added
c               when storage is poss.-os/2)
c
      r=n
      rtest=r/50
c      nreps=(r/50)-mod(r/50,1)
        nreps=(rtest)-mod(rtest,1.)
      if(rtest.ne.nreps)nreps=nreps+1
      m1=-49
      ip1=-99
c
      do 132 kkk=1,nreps
         m1=m1+50
         m2=50*kkk
         ip1=ip1+100
         ip2=kkk*100
         if(kkk.eq.nreps)then
           m2=n
           ip2=2*n-1
         endif
c         print*, 'yo ',m1,m2,ip1,ip2
c
C 700   M1=1
C       IP1=1
C         IF (N .GT. 50) GO TO 710
C      M2 = N
C       ISAIL=1
C       IP2=2*N-1
C        GOTO800
C 710   M2=50
C       ISAIL=2
C       IP2=100
C       GOTO800
C
CC        SET UP SECOND TIME
C 750   CONTINUE !PRINT751
CC       write(3,751)
C       ISAIL=1
C       M1=51
C       M2=N
C       IP1=101
C       IP2=2*N-1
CC       PRINT*,'M1=',M1,'M2=',M2
Cc
C SCOTT DANIELSEN 11-6-85
C CHANGED HICLUS TO WRITE OUT WORDS
C INSTEAD OF NUMBERS AT THE TOP OF THE
C DENDOGRAM.
C
C
  800 DO 501 NC=M1,M2
      write(71,713)words3(lk(nc))
      DO 500 NR=1,15
      WORD(NR,NC)=WORDS3(LK(NC))(NR:NR)
  713 format(a15)
      IF (WORD(NR,NC).EQ.' ')WORD(NR,NC)='.'
  500 CONTINUE
  501 CONTINUE
C
  997 if(m1.gt.50)call prntcon(12,3)       ! page eject
      DO 498 NR=1,15
      write(3,497)(word(nr,nc),nc=m1,m2)
C      if(m1.lt.50)write(3,497)(word(nr,nc),nc=m1,m2)
C      if(m1.gt.50)write(3,499)(word(nr,nc),nc=m1,m2)
  497 FORMAT(4X,50(1X,A1))
  499 FORMAT(5X,50(1X,A1))
  498 CONTINUE
C
C        PRINT FIRST DIGITS
c
C 800   DO551I=M1,M2
C 551   IP(I)=LK(I)/10
C       PRINT553,(IP(I),I=M1,M2)

C        PRINT SECOND DIGITS

C       DO552I=M1,M2
C 552   IP(I)=LK(I)-10*IP(I)
C       PRINT553,(IP(I),I=M1,M2)
C       PRINT554

C        INITIALIZE THE P ARRAY BY FILLING WITH BLANKS AND PERIODS

C        THE PERIODS REPRESENT VARIABLES AS YET UNCLUSTERED -- THE
C             BLANKS ARE FOR SPACING

       DO560I=1,599,2
      P(I+1)=BLANK
 560  P(I)=PERIOD

      NC=1
C     INDEX THROUGH THE MERGES, USIGG I

       DO600I=1,N1
C        COMPUTE THE APPROPRIATE PLACES IN THE P ARRAY

      JI=2*LKI(I)-1
      JJ=2*LKJ(I)-1

C        F+LL IN XS BETWEEN JI AND JJ IN THE P ARRAY

       DO590KK=JJ,JI
 590  P(KK)=XXXX

C        IF NEXT ONE IS CLUSTERED AT SAME LEVEL, DONT PRINT YET

      IF ((XX(I) .EQ. XX(I+1)) .AND. (I .NE. N1)) GO TO 600

C        PRINT OUT THE P ARRAY

 595  CONTINUE !PRINT596,XX(I),(P(III),III=IP1,IP2)
c      write(3,596)xx(i),(p(iii),iii=ip1,ip2)
      write(3,596)(p(iii),iii=ip1,ip2)        !take out side numbers
 596  format(5x,100a1)
c 596   FORMAT(1X,E16.8,4X,100A1)

 600   CONTINUE
C
 132   continue


C C        GO TO(900,750),ISAIL
C 900   CONTINUE!PRINT901
c       write(3,901)
c
c      catexp2.for goes here.
c
C        GOTO NEXT METHOD -- IF DONE, GO READ ANOTHER N

C      K=K+1
       IF (K .EQ. 3)then
         print*, ' stopped in hiclus'
         stop   ! GO TO 10
       endif
C      GO TO 50

 11    FORMAT(I3,I2)
 12    FORMAT(11A4)
 61    FORMAT(20HCONNECTEDNESS METHOD//)
 62    FORMAT(15HDIAMETER METHOD//)
 751   FORMAT(9HCONTINUED)
 553   FORMAT(20X,50(1X,I1))
 554   FORMAT(//)

 901   FORMAT(//,14H END OF METHOD)
       call prntcon(18,3)          !end condensed printing
       return
       end
c
      subroutine net(nepi,nuniqe,ncycle,ntepi,isnet,*)
      parameter (msize=5000)
      common /in/ win,ncon
      common /alpha/labels
      common /active/ av
      common /vector/ext
      common /parm/thresh,rf,h,log,nfunc,pdisk
      common /locate/isinc,iwords,niwords,iself,nmethod

      real*4 ext(msize),win(msize,msize),av(msize,5)
      character*15 labels(60000),pdisk*28,iwords(200)
      logical log,isnet,isinc,iself

c      data thresh/0.0/,rf/.1/,h/.001/,log/.true./
c
      ncon=nuniqe


      do 1 i = 1,ncon
      do 1 j = 1,ncon
      win(i,j)=0.0
    1 continue
c      call random
c      call norm
  20  call zero(ncon)
      call ccount(nepi,ncon,isnet,ntepi,*55,nold,iself)
      call cycle(thresh,rf,EXT,NCYCLE,h,LOG,nfunc,nold,nmethod)
c      call cycle(thresh,rf,EXT,NCYCLE,h,LOG,nepi,ntepi,nfunc,x) ! ** debug
      go to 20
  55  call squash   ! ** debug attempt to take log of win **
      call output(labels)
      return 1
      end
c  ______________________________Subroutine Norm__________________

      Subroutine norm
      parameter(msize=5000)
      common /in/ win,ncon

      real*4 win(msize,msize)

c         print67
c67    format('+ in norm')


c subtract the grand mean from every entry to center on zero...
      SUM=0.

      do 5 i=1,ncon
      do 5 j=1,ncon
      sum=sum+win(i,j)
    5 continue
      if(sum.eq.0)go to 7
      sum=sum/(ncon*ncon)

   7  do 6 i=1,ncon
      do 6 j=1,ncon
      win(i,j)=win(i,j)-sum
    6 continue

c now divide every entry by the absolute value of the largest entry
c so that the largest absolute value will be 1
c then multiply by x to make the largest value x
c This subroutine has no effect whatever: that's amazing!
 
c           if(sum.gt.99)then	
c           big=0.
c           do 1 i=1,ncon
c           do 1 j=1,ncon
c           if(abs(win(i,j)).gt.abs(big))big=win(i,j)
c    1      continue

c           do 2 i=1,ncon
c           do 2 j=1,ncon
c           if(win(i,j).eq.0)go to 2
c           win(i,j)=win(i,j)/abs(big)
c           win(i,j)=x*win(i,j)/abs(big) ! bigger for more dims? 8/31/92  jw

c    2      continue
c           endif

      return
      end

c_________________________________Subroutine Cycle_________________

      Subroutine Cycle(thresh,rf,ext,ncycle,h,LOG,nfunc,nold,nmethod)
      parameter (msize=5000)
      common /in/ win,ncon
      common /active/ av
      common /casecnt/casecnt
      common sense,on,off

      logical learn,sense,log
        real*4 av(msize,5),win(msize,msize),ext(msize),anet
      CHARACTER*1 noy
      Dimension casecnt(msize)

      data anet/0./

c         print67
c67    format('+ in cycle')


c      analog values.
c
        learn=.true.

c    4   print*, ' Analog?'
c        read(5,12)noy
c   12 FORMAT(A1)
c


        noy = 'Y'     !for now sd 3-14-91

c set the activation values of the input variables
c and zero out the external inputs for the next cycle

      do 5 i=1,ncon
      av(i,1)=av(i,1)+ext(i)
      if(sense)go to 5

c     keep on clamping only if locate method is by id and neuron is an id
c     neuron

      if(nmethod.eq.1 .and. i.eq.nold)go to 5



      ext(i)=0.0

    5 continue

c
      do 1, i=2,ncycle+1
c      DO 10 II=1,1
       do 2 j=1,ncon
        do 3 k=1,ncon
c
c     This line implements Jordan's totally self-reflexive model
        if(j.eq.k)go to 3   ! WHY NOT LET IT SPEAK TO ITSELF?
c     it interacts with the momentum and restoration terms

        anet= anet+(win(j,K)*av(K,i-1))
    3   continue



       go to(13,14,15,16)nfunc

c__________________The activation function stolen from JSPOT_______________

   13    Av(j,i)=(1.0/(1.0+(2.7183**(-1.0*anet))))
     +  +(rf*av(j,i-1))
         go to 17

c__________________Sigmoid varying from -1 to +1 (p708, NN, Vol 3 No 6, 1990)

   14   Av(j,i)=(1-(2.7183**(-1.0*anet)))/(1+(2.7183**(-1.0*anet)))
     +  +(rf*av(j,i-1))
        go to 17
c__________________The hyperbolic tangent function (Neuralware Demo)_______

   15   Av(j,i)= ((2.7183**anet)-(2.7183**(-1.0*anet)))/
     +((2.7183**anet)+(2.7183**(-1.0*anet)))
     +  +(rf*av(j,i-1))
        go to 17
c__________________A simple linear function_________________________________
   16  av(j,i)=anet+(rf*av(j,i-1))

   17 continue


       IF(EXT(J).NE.0.)AV(J,I)=EXT(J) ! external constraints, as in life.
c       IF(EXT(J).NE.0.)AV(J)=EXT(J) ! external constraints, as in life.

        anet=0.

        if(noy.eq.'y'.or.noy.eq.'Y')go to 2

       if(av(j,i).ge.thresh)av(j,i)=ON
       if(av(j,i).lt.thresh)av(j,i)=OFF
c
c       if(av(j).ge.thresh)av(j)=1.0
c       if(av(j).lt.thresh)av(j)=0.0

    2  continue
       if(learn)then
c       call hebb(i,h,log,x)       !(i,h,log) ! * solely for debug 8/31/92 jw
       call hebb(i,h,log)       !(i,h,log)
       ELSE
       endif

    1 continue
      return
      end

C_______________________________subroutine Hebb_________________________

      SUBROUTINE HEBB(itime,h,log)
c      SUBROUTINE HEBB(itime,h,log,x)  ! debug get rid of me !!!!!!
c      SUBROUTINE HEBB(h,log)
      parameter(msize=5000)
      common /in/ win,ncon
      common /active/ av
      logical log     !,isdone
      real*4 win(msize,msize),av(msize,5)

c                 print67
c67    format('+ in hebb')

c
c       sd 3-15-91
c      added isdone,change,delta.  Delta is increment change in connection
c      for win(i,j).  Continue learning only if change > delta. It will get
c      stuck if it osscilates so clamping is on.
c
c      isdone = .true.
c      change = .01

      if(log)then  !
      xbar=.5      ! set up the mean
      else         ! so that the logistic
      xbar=0.      ! can take on positive and negative values
      endif        ! call only when logistic function is used

      do 1 i=1,ncon
      do 2 j=1,ncon ! changed initial index from 2 to 1 1/19/93 -- jw

c      win(i,j)=win(i,j)+((av(i,itime)-xbar)*(av(j,itime)-xbar)*h)
      win(i,j)=win(i,j)+((av(i,itime))*(av(j,itime))*h)
c      win(i,j)=win(i,j)+((av(i)-xbar)*(av(j)-xbar)*h)

c      delta=abs(twin-win(i,j))
c      if(delta.gt.change)then
c        isdone = .false.
c      endif
    2 continue
    1 continue
c       (x)   ! ** debug -- get rid of me!!!!!!!
      CALL NORM
c      if(isdone)print*, ' Boy, am I bored'
      return
      end
c_______________________________Subroutine Output________________________

      Subroutine output(labels)
      parameter (msize=5000)
      common /in/ win,ncon
      real*4 win(msize,msize)
      Character*15 labels(60000)
c
      write(17,44)(labels(i),i=1,ncon)
  44  format(a15)
      write(16,35)
  35  format('(6f12.6)')
      DO 30 I=1,NCON
      WRITE(16,36)(WIN(I,J),J=1,NCON)
   36 format (6f12.6)
   30 continue
c
      return
      end


c
      subroutine zero(ncon)
      parameter (msize=5000)
      common /vector/ext
      common /active/ av
      real*4 ext(msize),av(msize,5)
      do 200 i=1,NCON
      EXT(I)=0.
      do 200 j=1,5
      av(i,j)=0.0
  200 continue
      return
      end



      subroutine coprint(win,ncon,nwds,lines,tit)
      parameter (msize=5000)
      common /dfile/ofile
      common /alpha/words
      real*4 win(msize,msize)
      character*15 words(60000),tit*80,ofile*80
      integer aindex(msize)

      write(20,66)tit,ofile
  66  format('TITLE:     'A69/'DATA FILE: ',a69//)
      write(20,6)nwds,ncon,lines
    6 FORMAT('TOTAL WORDS         ',I6/
     +       'TOTAL UNIQUE WORDS   ',I6/
     +       'TOTAL LINES          ',I7//)


      do 101 i=1,ncon
  101 aindex(i)=i
c
c sort aindex into alpahbetic order using asword as guide.
c
  202 flip=0
      do 203 i=1,ncon-1
      if(words(aindex(i)).le.words(aindex(i+1)))go to 203
      flip=1
      it=aindex(i)
      aindex(i)=aindex(i+1)
      aindex(i+1)=it
  203 continue
      if(flip.ne.0)go to 202

      do 10 i = 1,ncon
      do 10 j = 2,ncon

      if(aindex(i).eq.aindex(j))go to 10
c      if(words(aindex(i)).eq.words(j))go to 10
      if(win(aindex(i),aindex(j)).eq.0) go to 10
      num=aint(win(aindex(i),aindex(j)))
      write(20,100)words(aindex(i)),words(aindex(j)),
     +num
   10 continue

  100 format(a15,2x,a15,2x,i4)
      return
      end

      subroutine exclude
      common/del/nwrds,Bwords
      common /locate/isinc,iwords,niwords,iself,nmethod

      logical isinc,iself

      character*15 bwords(1000),iwords(200)
      nwrds=1

      if(isinc)then
        do 11 i = 1,100
        iwords(i)=' '
        read(22,100,end=2,err=2)iwords(i)
  11  continue
      endif

  2    niwords=i-1
      do 10 i=1,1000
      bwords(nwrds)=' '
      read(1,100,end=88)bwords(nwrds)

      if(isinc)then
      do 12 j = 1,200
   12 if(bwords(nwrds).eq.iwords(j))go to 10
      endif

      nwrds=nwrds+1
   10 continue

 100  format(a15)
   88 close(1)
      return
      end




      integer * 4 function icharint(anum,len)
c
c     this will convert an integer in 'character' form into an integer.
c
      character*10 anum
      len=index(anum,' ')-1

c      do 88 i = 1,len
c      num=index('0123456789',anum(i:i))
c      if(num.eq.0)then
c        print*, ' improper input ',anum(i:i),num
c      endif
c  88  continue

      icharint=0
      do 9 j=1,len
         icharint = icharint + (ichar(anum(j:j))-48)*10**(len-j)
 9    continue
      return
      end

      subroutine progress(message,npct)

      character*33 message*(*)

      write(*,100)npct,message
 100  format('+ In progress',i4,2x,a30)
      return
      end

      subroutine status(message)

      character*33 message*(*)

      print*, message

      return
      end


      subroutine choicewin(nx,ny,end,nchoice,darray,nstep,nselect,*,*)

      character*(*)darray(10)

   1  do 10 j=1,nchoice
  10  print*,j,') ',darray(j)
      print*, ' Please make a selection'
c     make selection
      read(*,*,err=1)nselect
      return
      end

      subroutine error(ertype,nerr,square,
     +indent,npos,text,ntab,*,message)

      character*(*)text,message
          character*80 tline
      character*1 key !integer*2 getche
      integer ertype

      go to (2,2)ertype

 1    if(message.eq.' File descriptor')then
         tline='Error using '//message
      else
         tline='Error opening '//message
      endif
      go to 3

 2    tline=' Sorry, I can''t use: ' // message

 3    continue
      print*, tline
      print*, ' Please re-enter'
      return 1
      end

      subroutine reset(nlines,ntepi,nwrds,nuniqe,nwords)
      parameter (msize=5000)

      common /freq/kwords
      common /alpha/words
      common /vector/ext
      common /in/win,ncon
      common /casecnt/casecnt

      real*4 ext(msize),win(msize,msize),casecnt(msize)

      integer kwords(60000)
      character*15 words(60000)

      nlines=0
      ntepi=0
      nwrds=0
      ncon=0
      nwords=0
      nuniqe=0

      do 33 i = 1,60000
        words(i)=' '
        kwords(i)=0
 33   continue

      do 44 i = 1,msize
      ext(i)=0.
      casecnt(i)=0.
      do 44 j = 1,msize
 44   win(i,j)=0.0
      return
      end

      subroutine prints(eps,tuwds,lines,twds,tit,isnet,ncycle,clamp,
     +isfreq)
      parameter (msize=5000)
      common /dfile/ofile
      common /freq/kwords
      common /alpha/words
      common /window/iswin,winsize,winwrds,nslide
      common /parm/thresh,rf,h,log,nfunc,pdisc
      common /casecnt/casecnt
      logical iswin,isnet,isfreq
c
      character*80 tit,ofile
      character*28 pdisc
      character*15 clamp*3,words(60000),ait
      integer twds,tuwds,eps,lines,winsize,winwrds,kwords(60000)
      integer aindex(60000)
      real casecnt(msize)
c
c     prints out stats for words.  Thanks once again to the mighty RA.

c     first write out header info

      write(3,66)tit,ofile
  66  format('TITLE:     ',A69/'DATA FILE: ',A69//)

      if(isfreq)then
      write(3,56)twds,tuwds,lines

   56 FORMAT('TOTAL WORDS         ',I7/
     +       'TOTAL UNIQUE WORDS   ',I7/
     +       'TOTAL LINES          ',I7//)
      go to 566
      endif

       if (.not.isnet)then
        if(.not.iswin)write(3,6)twds,tuwds,'EPISODES',eps,lines
        if(iswin)write(3,6)twds,tuwds,'WINDOWS ',eps,lines
    6 FORMAT('TOTAL WORDS           ',I7/
     +       'TOTAL UNIQUE WORDS   ',I7/
     +       'TOTAL ',A8,  '       ',I7/
     +       'TOTAL LINES          ',I7)
      else
        if(.not.iswin)write(3,17)twds,thresh,tuwds,rf,
     +   'EPISODES',eps,ncycle,lines,pdisc
        if(iswin)write(3,17)twds,thresh,tuwds,rf,
     +   'WINDOWS',eps,ncycle,lines,pdisc
      endif

  17  FORMAT('TOTAL WORDS          ',I7,5X,'THRESHOLD',11x,F5.3/
     + 'TOTAL UNIQUE WORDS   ',I7,5X,'RESTORING FORCE',5x,F5.3/
     + 'TOTAL ',A8,  '       ',I7,5X,'CYCLES',12x,i5/
     + 'TOTAL LINES          ',I7,5x,'FUNCTION',12X,A28)

      if(.not.iswin.and.isnet)write(3,76)clamp
  76   format(30x,'CLAMPING              ',A3)
C

      if(iswin.and..not.isnet)write(3,77)winsize,nslide
  77  format('WINDOW SIZE             ',I4/
     +       'SLIDE SIZE              ',I4)
C
      if(iswin.and.isnet)write(3,78)winsize,clamp,nslide
  78   format('WINDOW SIZE             ',I4,5x,'CLAMPING',12x,a3/
     +       'SLIDE SIZE              ',I4)
      write(3,79)
  79  format(/)

c     build index for alaphabetic sorts


 566   do 501 i=1,tuwds
  501  aindex(i)=i

c     sort words into decending order

      if(isfreq)then
       npct=0
 502   flip = 0
      do 503 i=1,tuwds-1
        if(kwords(i).ge.kwords(i+1))go to 503
        flip=1
        it=kwords(i)
        kwords(i)=kwords(i+1)
        kwords(i+1)=it
        ait=words(i)
        words(i)=words(i+1)
        words(i+1)=ait
  503 continue
      if(npct.ge.98)npct=0
c      call progress('Descending, Dante',npct)
      npct=npct+2
      if(flip.ne.0)go to 502
      endif

c sort aindex into alphabetic order.

c      call progress(' ',100)
      npct=0

  602 flip=0
      do 603 i=1,tuwds-1
      if(words(aindex(i)).le.words(aindex(i+1)))go to 603
      flip=1
      it=aindex(i)
      aindex(i)=aindex(i+1)
      aindex(i+1)=it

  603 continue
      if(npct.ge.98)npct=0
c      call progress('  Alphabetizing',npct)
      npct=npct+2
      if(flip.ne.0)go to 602

c output sorted lists with headings

      ipage=1
      
513   continue     

c 513  call progress(' ',100)

      if(isfreq)then
      write(3,500)
      else

      write(3,300)
      endif

  500 format(T5,'DESCENDING FREQUENCY LIST',
     +T46,'ALPHABETICALY SORTED LIST'//
     +3X,'INDEX',2X,'FREQ.',2X,'PCENT.',2X'WORD',15X
     +,'INDEX',2X,'FREQ.',2X,'PCENT.',2X,'WORD'/
     +3X,'-----',2X,'-----',2X,'------',2X,'---------------',4X
     +,'-----',2X,'-----',2X,'------',2X,'---------------'/)

  300 format(T5,'DESCENDING FREQUENCY LIST',
     +T46,'ALPHABETICALY SORTED LIST'//

     +27X,'CASE',1X,'CASE',32X,'CASE',1X,'CASE'/
     +'WORD',13x,'FREQ',1X,'PCNT',1X,'FREQ',1X,'PCNT'
     +,5X,'WORD',13X,'FREQ',1X,'PCNT'1X,'FREQ',1X,'PCNT'/
     +,'---------------',1x,4(1x,'----'),5x,'---------------',1x
     +,4(1x,'----'))

      nprint=22
      if(iswin)nprint=24
      if(isfreq)nprint=21

c
      do 57 i=1,tuwds
c      if(isfreq)then
        pcent1=(real(kwords(i))/real(twds))*100.
        pcent2=(real(kwords(aindex(i)))/real(twds))*100.
      if(isfreq)then
        write(3,58)i,kwords(i),pcent1,words(i),
     +  aindex(i),kwords(aindex(i)),pcent2,words(aindex(i))
   58 format(1X,I5,2X,I5,4X,f5.2,2X,A15,3X,I5,2X,I5,4X,f5.2,2X,A15)
      else
        write(3,8)words(i),kwords(i),pcent1,int(casecnt(i))
     +  ,casecnt(i)/eps*100.
     +  ,words(aindex(i)),kwords(aindex(i)),pcent2
     +  ,int(casecnt(aindex(i))),casecnt(aindex(i))/eps*100.

  8   format(a15,2x,i4,1x,f4.1,1x,i4,1x,f4.1,
     +5x,a15,2x,i4,1x,f4.1,1x,i4,1x,f4.1)
      endif

c     paginiation

      nprint=nprint+1
      if(nprint.le. 60)go to 57
      if(i.eq.tuwds)go to 588
      call prntcon(12,3)         !eject page
      nprint=9
      ipage=ipage+1
      write(3,801)ipage
  801 format('  Page ',i2/)
      if(isfreq)then
      write(3,500)
      else
      write(3,300)
      endif
   57 continue

c     eject page

588   call prntcon(12,3)      !eject page
      close(23)

      return
      end


c     Utilities

c
      subroutine charreal(anum,rreal)

c     another function of limited functions.  Will accept a character
c     real number and return a real real number.  Recognizes sign as well.

      character*10 anum,first,second
      integer*4 icharint

      logical isneg
      isneg=.false.
      if(anum(1:1).eq.'-')then
        isneg=.true.
        anum=anum(2:10)
      endif
      do 10 i = 1,10
        if(anum(i:i).eq.'.')then
          first=anum(1:i-1)
          second=anum(i+1:10)
          go to 1
        endif
  10  continue
      num1=icharint(anum,len)
      rreal=real(num1)
      if(isneg)rreal=0.-rreal
      return
  1   num1=icharint(first,len1)
      num2=icharint(second,len2)
      rnum2=real(num2)/10**len2
      rreal=num1+rnum2
      if(isneg)rreal=0.-rreal
      return
      end

c      subroutine timer(ntime)
c      integer  hour,minute,second,hund
cc
c      call gettim(hour,minute,second,hund)
c      ntime=(hour*60*60)+(minute*60)+second
c      return
c      end
      subroutine checknum(anum,*)
      character*(*) anum
      len=index(anum,' ')-1

      do 88 i = 1,len
      num=index('0123456789',anum(i:i))
      if(num.eq.0)return 1
  88  continue
      return
      end
      subroutine checkreal(anum,*)
      character*(*) anum
      len=index(anum,' ')-1

      do 88 i = 1,len
      num=index('0123456789.',anum(i:i))
      if(num.eq.0)return 1
  88  continue
      return
      end

c
c      subroutine catwrt(prog,vers,unit,num)
c
c     writes date time program and version to file
c     with printer control characters or not.
c
c      character*(*) prog,vers
c      character*1 cc,esc
c      integer  hour,minute,second,hund,year,month,day,unit
c      call getdat(year,month,day)
c      call gettim(hour,minute,second,hund)
c      year = year-2000
c      if(num .eq.1)then
c      write(unit,1000)prog,vers,month,day,year,hour,minute,second
c 1000 format(/////////5x,a10,5x,a5,
c     +15x,i2.2,'/'i2.2,'/',i2.2,6x,i2.2,':'i2.2,':',i2.2)
c      return
c      endif
c      esc=char(27)
c      cc=char(14)
c      write(unit,100)esc,cc,prog,vers
c      call prntcon(20,3)
c      write(unit,101)month,day,year,hour,minute,second
c 100  format(2a1,10x,a10,2x,a5/)
c 101  format(27x,i2.2,'/'i2.2,'/',i2.2,6x,i2.2,':'i2.2,':',i2.2//)
c      return
c      end
c
       subroutine prntcon(ncc,nunit)
       CHARACTER*1 cc,esc
c
c      this will send a control code to the printer with
c      esc before it.  Based on simple ASCII control codes
c      should work with any printer (ha ha).
c
       esc=char(27)
       cc=char(ncc)
       write(nunit,313)esc,cc
  313 format(2A1)
      return
      end

c Help section

c    Parameter routines

      subroutine freqparms(tit,freqdef,*)

      common/videoconfig/vc
      common /xycoord/xy
      common /dfile/ofile
      logical freqdef
      character*80 tit,dfile,efile,ofile

      print*, ' Please enter a title for the run'
      print*, ' '
      read(*,100,end=99)tit
100   format(a80)

  1   Print*, ' Enter name of data file'
      print*, ' '
      read(*,100,end=99)dfile
      open(unit=2,file=dfile,status='old',err=1,recl=1000)  ! data file

  2   Print*, ' Enter name of your exclude file'
      print*, ' (just press enter for default)'
      print*, ' '
      read(*,100,end=99) efile
      open (unit=1,file=efile,status='old',err=2,iostat=ic) !exclude file


  3   print*, ' Enter name of your output file'
      print*, ' '
      read(*,100,end=99) ofile
      open(unit=3,file=ofile,status='unknown',err=3)        !output file

      open(unit=14,file='gtridof.me',status='unknown',err=3)      ! scratch file
c    +,blocksize=4096)   ! I killed this part of the open 10/21/15  jw

      return

 99   return 1   ! want to go home

      end

c here is where we open the co file

c      nerr=8
c      if(texte(8).ne.' ')then
c       isco=.true.
c       if(isco)open(unit=20,file=texte(8),status='unknown'     !matrix file
c     + ,err=3)
c      else
c       isco=.false.
c      endif
c
c      ofile=texte(2)
c      return
      subroutine crdmake

C PROGRAM TO TRANSPOSE A MATRIX
      parameter (msize=5000)

      common /in/win,ncon
      common /alpha/labels

      real*4 win(msize,msize)
      character*15 labels(60000)
      data ndim/6/  ! ** debug jw 11/18/15  **

      open(unit=77,file='delme.win')
      do 25 i=1,ncon     ! we need an extra copy of this fellow
      do 25 j=1,ncon     ! since both crdmake and hiclus will
   25 write(77,777)win(i,j)
  777 format (6f12.6)

   21 CALL BIGEIG(NCON,ncon,win,labels)

      rewind(77)
      do 26 i=1,ncon     ! we need an extra copy of this fellow
      do 26 j=1,ncon     ! since both crdmake and hiclus will
  26  read(77,777)win(i,j)
      close(77,status='delete')
      end

        SUBROUTINE BIGEIG(NCON,NDIM,Z,labels)!,ncon)

      parameter (msize=5000)
C
C Joseph Woelfel
C Copyright 1984

C All Rights Reserved
C
C CALCULATES NDIM EIGENVECTORS FROM AN NCON BY
C NCON MATRIX OF SCALAR PRODUCTS OR COVARIANCES
C
C THIS PROGRAM IS NOT READY FOR PRIME TIME...
C
      DIMENSION Z(msize,msize),VT(msize),C(msize),ITER(msize)
     +,fax(msize,msize),ROOT(msize),vec2(msize)
      real ptot(msize)
      character*15 labels(60000)
      character*1 icc  ! added 10/21/15 jw
      integer cutoff

      DATA T/.001/,MAXITR/5000/,LFNO/6/,LFNP/9/
      
c ** now a little trick to embiggen the numbers ** ! 11/18/2015 jw

        do 1 i=1,ncon
        do 1 j=1,ncon
        z(i,j)=100*z(i,j)
1     continue      

c** end of little trick   11/18/2015      jw   
      
C
  601 DO 150 KOUNT=1,NDIM

C         ITERATE TO EQUIVALENCE PRODUCT VECTORS


   26 DO 40 J=1,NCON
      VT(J)=1.0
   40 CONTINUE

      ITER(KOUNT)=1
C
C SAVE OLD TOLERANCE IN CASE OF MAXITR LIMIT ALTERATIONS
      OLDT=T
C
  120 DO 50 M=1,NCON
      C(M)=0.0
      DO 50 N=1,NCON
      C(M)=C(M)+Z(M,N)*VT(N)
   50 CONTINUE

      ITER(KOUNT)=ITER(KOUNT)+1

      DIV=C(1)
      DO 60 J=2,NCON
      IF(ABS(C(J)).GT.ABS(DIV))DIV=C(J)
   60 CONTINUE

      IF(DIV.NE.0)GO TO 9
      WRITE(LFNO,10) KOUNT
   10 FORMAT('  CANNOT FACTOR SPACE   SUM OF ELEMENTS IN ',I3,' RO'
     +'W ARE 0')
      STOP
C
9     CONTINUE
      DO 70 K=1,NCON
      CTMP=C(K)/DIV
      IF(ABS(CTMP-VT(K))-T)70,70,90
   70 VT(K)=CTMP
      GO TO 110
   90 DO 1231 L=K ,NCON
      VT(L)=C(L)/DIV
 1231 CONTINUE

C
C  CHECK THE ITERATIONS AGAINST MAXITR FOR THIS ROOT...REDUCE T IF NEE

1230  IF(ITER(KOUNT).LE.MAXITR)GO TO 120
      T=T*10.
c      IF(T.GE.1) GO TO 4184
            IF(T.GE.100) GO TO 4184 ! raised tolerance 1 order of magnitude 26 Jan 2016 jw
      ICC=' '
      WRITE(LFNO,1947) ICC,KOUNT,T
1947  FORMAT(A3,'MAXITR REACHED ON ROOT ',I2,' TOLERANCE REDUCED TO',
     +F13.9)
      ITER(KOUNT)=1
      GO TO 120
4184  WRITE(LFNO,4176)
4176  FORMAT('  A TOLERANCE OF 100 IS NOT NORMALLY ACCEPTABLE. '/ ! changed to 10 jw
     +'  CHECK YOUR INPUT')
      STOP
C
C         RETRIEVE ROOT AND NORMALIZE VECTOR
C

  110 DIV=SIGN(ABS(DIV),DIV)
      ROOT(KOUNT)=DIV
      SUMC=0.0

      DO 130 K=1,NCON
      SUMC=SUMC+VT(K)**2
  130 CONTINUE

      P=SQRT(ABS(SUMC))
      Q=SQRT(ABS(DIV))

      DO 140 K=1,NCON
      FAX(K,KOUNT)=(VT(K)/P)*Q
  140 CONTINUE
C
C RESTORE THE TOLERANCE TO THE PARAMETER VALUE
      T=OLDT
C
C        COMPUTE RESIDUAL MATRIX
C
      SGN=1
      IF(DIV.GT.0)SGN= -SGN
      DO 150 K=1,NCON
      DO 150 L=1,NCON
      Z(K,L)=Z(K,L)+SGN*(FAX(K,KOUNT)*FAX(L,KOUNT))
  150 CONTINUE

C          SORT VECTORS FROM NATURAL TO DESCENDING ORDER
      NUML1=NCON-1
 4000 NDEX=0
C
      DO 3000 IXE=1,NUML1
      IF (ROOT(IXE).GE.ROOT(IXE+1)) GO TO 3000
      NDEX=1
      TMP=ROOT(IXE)
      ROOT(IXE)=ROOT(IXE+1)
      ROOT(IXE+1)=TMP
      ITMP=ITER(IXE)
      ITER(IXE)=ITER(IXE+1)
      ITER(IXE+1)=ITMP
      DO 2050 J=1,NCON
      TMP=FAX(J,IXE)
      FAX(J,IXE)=FAX(J,IXE+1)
 2050 FAX(J,IXE+1)=TMP
 3000 CONTINUE
      IF(NDEX.NE.0)GO TO 4000
C
C         COMPUTE CUMULATIVE PERCENTAGES OF DISTANCE
C
C KEEP RECORD OF THE 'ZERO-ROOT', AS THIS DETERMINES
C THE IMAGINARY PART OF THE SPACE
C
  811  RMIN=ABS(ROOT(1))
      STOT=ROOT(1)
      CUTOFF=1
      DO 525 K=2,NCON
      STOT=STOT+ROOT(K)
      IF(ABS(ROOT(K)).GT.RMIN)GO TO 525
      RMIN=ABS(ROOT(K))
      CUTOFF=K
  525 CONTINUE
      SREAL=0
      CUTOFF=CUTOFF-1
      DO 1806 K=1,CUTOFF
      SREAL=SREAL+ROOT(K)
 1806  CONTINUE
      SIMAG=STOT-SREAL
      WARP=SREAL/STOT
C      DIM(INCMNT)=CUTOFF
C
      DO 530 K=1,NCON
      PTOT(K)=     ABS  ((ROOT(K)/STOT)*100.0)
  530 CONTINUE
C
      DO 64 I=1,CUTOFF
      VT(I)=ABS(ROOT(I)/SREAL*100.)
  64  CONTINUE
      IF(CUTOFF+1.GT.NCON)GO TO 65
      ICT=CUTOFF+1
      DO 66 I=ICT,NCON
      if(simag.eq.0)go to 66
      VT(I)=ABS(ROOT(I)/SIMAG*100.)
   66 CONTINUE
  65  CONTINUE
C
      ZZERO = 0.0
      DO 790 K=1,NCON
790   CCUBE=ZZERO+VEC2(K)**3
      DO 793 K=1,NCON
793   IF(CCUBE.LE.0.0) VEC2(K)=-VEC2(K)

C ALL DONE, WRITE IT OUT AND GO HOME...
C
c      print *, '                Eigenvalues'
c      WRITE(3,700)
c 700  FORMAT('0',30X,'Eigenvalues',/)
c      do 5904 i=1,ndim
c      WRITE(LFNO,7)i,ROOT(I)
c    7 FORMAT(' Root ',i2,'=,'f15.2)
c 5904 continue
c      print *, 'Warp Factor = ', warp

      WRITE(LFNP,5903)ncon,cutoff,NDIM
c      print*,' ncon, cutoff, ndim = ',ncon,cutoff,ndim  ! debug
 5903 FORMAT('(6F12.4)',2X,3I3)        !srd 6/4/92  change to 3i3 11/19/15 6f12.4
      DO 400 K=1,ncon
      WRITE(LFNP, 270)  (FAX(K,J),J=1,NDIM)
  270 FORMAT (6F12.4)  ! ** jw 11/18/15 **
  400 CONTINUE
      do 401 i=1,ncon
c      print*,' Writing the labels.... i= ',i ! debug
      write(lfnp,402)labels(i)
  402 format(a15)
  401 continue
c  404 continue
      return
      END
      subroutine yorn(*,*)
c
      character*1 ans
    1 write(*,'(a,/)') ' ====>'
      read(*,2,end=3)ans
    2 format(a1)
      if(ans.eq.'Y'.or.ans.eq.'y')return 1
      if(ans.eq.'N'.or.ans.eq.'n')return2
      write(*,4)
    4 format('  Please answer Yes or No.')
      go to 1
    3 write(*,5)
    5 FORMAT(' PLEASE, answer the question.')
      go to 1
      end

      subroutine netparms(tit,nlim,isnet,ncycle,clamp,netdef,*)



      common /window/iswin,winsize,winwrds,nslide
      common /parm/thresh,rf,h,log,nfunc,pdisc
      common /locate/isinc,iwords,niwords,iself,nmethod

      common /dfile/ofile  ! to print output file in prints

      common sense,on,off

      character*64 fontpath
      character*30 list
      character*(10) option (6)
      character*15 iwords(200)
      character*80 tit,ofile
      character*17 message(19)/' Run title',' Data file',
     +                         ' Exclude file',' Include file',
     +                         ' File descriptor',' Unique words',
     +                         ' Window size',' Slide size',
     +                         ' Cycles',' Clamping',' Zelf anlysis',
     +                         ' Function Form', ' Threshold',
     +                         ' Decay Rate', ' Learning Rate',
     +                         ' Catpac file', ' Label file',
     +                         ' Win file',' Coordinate file'/

      character*80 hfile(15)/'galileo\help\cp_htit.hlp',
     +                       'galileo\help\cp_htxt.hlp',
     +                       'galileo\help\cp_exclu.hlp',
     +                       'galileo\help\cp_inclu.hlp',
     +                       'galileo\help\cp_fdesc.hlp',
     +                       'galileo\help\cp_nuniq.hlp',
     +                       'galileo\help\cp_wnsze.hlp',
     +                       'galileo\help\cp_hslde.hlp',
     +                       'galileo\help\cp_ncycl.hlp',
     +                       'galileo\help\cp_clamp.hlp',
     +                       'galileo\help\cp_self.hlp',
     +                       'galileo\help\cp_func.hlp',
     +                       'galileo\help\cp_trsh.hlp',
     +                       'galileo\help\cp_dcay.hlp',
     +                       'galileo\help\cp_srat.hlp'/

      character*80 texte(19)  /' ',' ','catpac/exclude.dat'
     +                        ,' ',' ','50','7','1','1','Yes','None'
     +                        ,'Sigmoid (-1 - +1)','.0','.9','.001',
     +                        ' ',' ',' ',' '/
      character*3 clamp,fdisc(4)*28,pdisc*28
      character*4 aend(4)/'.cat','.lbl','.win','.crd'/
      character*25 fself(3)/'None','Locate by single ID',
     +                      'Locate by self-reference'/

      integer winsize,winwrds
c      integer*2 icharint*4    !  ** debug ** jw 10/21/15
      logical iswin,isinc,iself,isnet,sense,log,fnet,
     +ischng/.false./,netdef

       data fdisc/'Sigmoid (0 - +1)','Sigmoid (-1 - +1)',
     +'Hyberbolic Tangent (-1 - +1)','Linear (-1 - +1)'/
      data nfunc/2/,nself/1/

      if(netdef)then
        nfunc=2
        nself=1
        netdef=.false.
        texte(3)='/galileo/runner/exclude.dat'
        texte(6)='50'
        texte(7)='7'
        texte(8)='1'
        texte(9)='1'
        texte(10)='Yes'
        texte(11)='None'
        texte(12)='Sigmoid (-1 - +1)'
        texte(13)='.0'
        texte(14)='.9'
        texte(15)='.001'
       endif

      isnet=.true.

c     end init var setup

  1   write(*,100)(i,message(i),texte(i),i=1,19)
 100  format(1x,i2,2x,a17,2x,a30)
      print*,'Enter number of option to change,-1 to run, 0 to end'
      read(*,101,end=22)lopt
c      print*,' lopt = ', lopt  ! ** debug **
      
      if(lopt.eq.0)go to 99
      if(lopt.eq.-1)go to 22
 101  format(i2)

      if(lopt.eq.11.and.nkey.eq.13)then
        call choicewin(indent,npos,square-1,3,fself,nstep,nself,*22,*99)
        texte(ntab)=fself(nself)
      endif

      if(lopt.eq.12.and.nkey.eq.13)then
        call choicewin(indent,npos,square-1,4,fdisc,nstep,nfunc,*22,*99)
        texte(ntab)=fdisc(nfunc)
      endif

      print*, ' Enter ',message(lopt)
      read(*,103)texte(lopt)
 103  format(a80)



c     create file names if ntab is on file descriptor field

      if(lopt .eq.5 .and. texte(lopt).ne.' ')then
       len=index(texte(lopt),' ')-1
       do 66 j = 1,4
  66   texte(15+j)=texte(lopt)(1:len)//aend(j)
      endif


c     end create file names

      if(texte(13).ne.'.0'.and.texte(13).ne.'.5')ischng=.true.
      if(nfunc .eq.1 .and. .not. ischng)texte(13)='.5'
      if(nfunc .ne.1 .and. .not. ischng)texte(13)='.0'

      go to 1

c     run the job
c 22		print*,' Well, well, well!'
 22   continue

c     first, do conversions

      tit=texte(1)
      pdisc=fdisc(nfunc)
      isinc=.false.
      if(texte(4).ne.' ')isinc=.true.

      nerr=6
      call checknum(texte(6),*98)
      nlim=icharint(texte(6)(1:10),len)

      if(texte(7).eq.'-1')then
        iswin=.false.
        winsize=-1
      else
      nerr=7
      call checknum(texte(7),*98)
        iswin=.true.
        winsize=icharint(texte(7)(1:10),len)
      endif

      nerr=8
      call checknum(texte(8),*98)
      nslide=icharint(texte(8)(1:10),len)

      nerr=9
      call checknum(texte(9),*98)
      ncycle=icharint(texte(9)(1:10),len)

      if(texte(10)(1:1).eq.'y'.or.texte(10)(1:1).eq.'Y')then
        clamp='Yes'
        sense=.true.
      else
        clamp='No'
        sense=.false.
      endif

      if(nfunc.eq.1)then
        off=0.
        log = .true.
      else
        log=.false.
        off=1.
      endif
c
      if(nself.ne.1)then
        iself=.true.
        nmethod=nself-1
      else
        iself=.false.
        nmethod=0
      endif

      nerr=13
      call checkreal(texte(13),*98)
      call charreal(texte(13)(1:10),thresh)

      nerr=14
      call checkreal(texte(14),*98)
      call charreal(texte(14)(1:10),rf)
      rf=1.-rf

      nerr=15
      call checkreal(texte(15),*98)
      call charreal(texte(15)(1:10),h)

       nerr=5
       if(texte(5).eq.' ')go to 11

c      print*, sense,off,log
c      print*, iswin,winsize,winwrds,nslide
c      print*, thresh,rf,h,log,nfunc,pdisc
c      print*, isinc,iself,nmethod
c      print*, tit,nlim,isnet,ncycle,clamp

c     end conversions

c     now open all files

      nerr=2
      if(texte(2).eq.' ')go to 11
      open(unit=2,file=texte(2),status='old',err=11,recl=1000) ! data file
      nerr=14
      open(unit=14,file='gtridof.me',status='unknown',err=11)   ! scratch file
c     +,blocksize=4096)  ! ** i got rid of this 10/21/15 jw
      nerr=2
      open (unit=1,file=texte(3),status='old',err=11)          !exclude file
      nerr=5
      open(unit=3,file=texte(16),status='unknown',err=11)      !catpac  file
c      print*,' ifile = ', texte(16)  ! ** debug **
      nerr=5
      open(unit=17,file=texte(17),status='unknown',err=11)     !label  file
      nerr=5
      open(unit=16,file=texte(18),status='unknown',err=11)     !win   file
      nerr=5
      open(unit=9,file=texte(19),status='unknown',err=11)      !coord   file
      nerr=4
      if(isinc)open(unit=22,file=texte(4),status='old',err=11) !include file

      ofile=texte(2)
      fnet=.false.
      return


 11   print*,nerr,texte(lopt),message(nerr)
         go to 1
c call error(1,nerr,square,indent,npos,texte(ntab),
c     +ntab,*1,message(nerr))

 98   print*, nerr,texte(lopt),message(nerr)
         go to 1

c ccall error(2,nerr,square,indent,npos,texte(ntab),
c     +ntab,*1,texte(nerr))

 99   fnet=.false.
      return 1  ! want to go home
      end

        subroutine squash
        
        parameter(msize=5000)
      common /in/ win,ncon

      real*4 win(msize,msize)
c	  print*,' Here we are in squash... '
      x=100
        do 1 i=1,ncon
        do 1 j=1,ncon
c	  win(i,j)=win(i,j)*10
c		print*,' win(3,j) = ', win(3,j)
        if(win(i,j).eq.0) go to 1
c	  print*,' win(i,j) before = ', win(i,j)
        if(win(i,j).lt.0)win(i,j)=log(x*(abs((win(i,j))-1)))*(-1)
        if(win(i,j).gt.0) win(i,j)= log(x*((win(i,j))+1))
c	  print*,' win(i,j) after = ',win(i,j)
c	  print*,'win(2,j) = ', win(2,j)
1     	continue
        return
        end
