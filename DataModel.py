class Region:
    def __init__(self,x1,y1,x2,y2,label=None,score=None):
        assert x1<=x2
        assert y1<=y2
        self.label = label
        self.score = score
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2
        self.w=x2-x1
        self.h=y2-y1
        self.p1 = (x1,y1)
        self.p2 = (x2,y2)
    def iou(self,b):
        '''
        :param b:Region
        :return: Intersection Over Union
        '''
        assert isinstance(b,Region)
        return self.getAreaOfIntersection(b)/float(self.getAreaOfUnion(b))
    def getIntersectedRegion(self,b):
        '''
        :param b:Region
        :return: Overlapping Rectangle
        '''
        assert isinstance(b, Region)
        xA = max(self.x1, b.x1)
        yA = max(self.y1, b.y1)
        xB = min(self.x2, b.x2)
        yB = min(self.y2, b.y2)
        xmin = min(xA,xB)
        xmax = max(xA,xB)
        ymin = min(yA,yB)
        ymax = max(yA,yB)
        return Region(x1=xmin,y1=ymin,x2=xmax,y2=ymax)
    def getAreaOfIntersection(self,b):
        '''
        :param b:Region
        :return: Area of Overlap
        '''
        # compute the area of intersection rectangle
        return self.getIntersectedRegion(b).getArea()
    def getAreaOfUnion(self,b):
        '''
        :param b: Region
        :return: Area of Union
        '''
        assert isinstance(b,Region)

        return self.getArea()+b.getArea()-self.getAreaOfIntersection(b)

    def getArea(self):
        return float(self.w+1) * (self.h+1)


