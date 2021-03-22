#include <xtensa/tie/xt_ivpn.h>
#include <xtensa/tie/xt_misc.h>
#include <xi_tile_manager.h>
#include <xi_intrin.h>
#include <xi_core_api.h>

typedef struct HaloParam {
    size_t sizes;
    unsigned short left;
    unsigned short top;
    unsigned short width;
    unsigned short height;

    unsigned short tempW;
    unsigned short tempH;
    unsigned char  tempMaxVal;

    unsigned short srcW;
    unsigned short srcH;
} HaloParam, *pHaloParam;

void GetPreComputeIdx(HaloParam param, uint16_t* reIdxVec[3]) 
{
    xb_vecNx16U *__restrict rHeightTable = NULL;
	xb_vecNx16U *__restrict rWidthTable  = NULL;
    xb_vecNx16U *__restrict rHeightSplit = NULL;

    xb_vecNx16U left = param.left;
	xb_vecNx16U top = param.top;
	xb_vecNx16U mulHBase  = param.tempH;
	xb_vecNx16U divHBase  = param.srcH;
	xb_vecNx16U mulWBase  = param.tempW;
	xb_vecNx16U divWBase  = param.srcW;
	xb_vecNx16  rem;
	xb_vecNx16U hBlock = IVP_SEQNX16U();
	xb_vecNx16U hBlockTmp; 	// 暂存乘法结果
	xb_vecNx16U wBlock = hBlock;
	xb_vecNx16U wBlockTmp;	// 暂存乘法结果
	xb_vecNx16U hBlockofs = IVP_ADDNX16U(hBlock, top);
	xb_vecNx16U wBlockofs = IVP_ADDNX16U(wBlock, left);

    int idx;
    xb_vecNx16U offset;
    for (idx = 0; idx < param.srcH; idx += XCHAL_IVPN_SIMD_WIDTH) {
        offset = idx;

        rHeightTable = OFFSET_PTR_NX16U(reIdxVec[0], 0, param.srcH, idx);
        hBlockTmp = IVP_ADDNX16U(hBlockofs, offset);
		hBlockTmp = IVP_MULNX16(hBlockTmp, mulHBase);
		IVP_DIVNX16(hBlockTmp, rem, hBlockTmp, divHBase);
        *rHeightTable = IVP_MULNX16(hBlockTmp, mulWBase);

        rHeightSplit = OFFSET_PTR_NX16U(reIdxVec[1], 0, param.srcH, idx);
        hBlockTmp = IVP_ADDNX16U(hBlock, offset);
        *rHeightSplit = IVP_MULNX16(hBlockTmp, mulWBase);
    }

    for (idx = 0; idx < param.srcW; idx += XCHAL_IVPN_SIMD_WIDTH) {
        offset = idx;

        rWidthTable = OFFSET_PTR_NX16U(reIdxVec[2], 0, param.srcW, idx);
        wBlockTmp = IVP_ADDNX16U(wBlockofs, offset);
        wBlockTmp = IVP_MULNX16(wBlockTmp, mulWBase);
        IVP_DIVNX16(*rWidthTable, rem, wBlockTmp, divWBase);
    }
}