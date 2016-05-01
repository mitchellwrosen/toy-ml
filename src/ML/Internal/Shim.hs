{-# LANGUAGE DataKinds      #-}
{-# LANGUAGE KindSignatures #-}

-- | Shim between @hmatrix@ and @linear@.

module ML.Internal.Shim
  ( pinv
  ) where

import Data.Vector  (Vector)
import GHC.TypeLits (KnownNat, Nat)

import qualified Data.Vector           as Vector
import qualified Data.Vector.Generic   as Vector.Generic
import qualified Linear.V              as Linear
import qualified Numeric.LinearAlgebra as HMatrix

type LMatrix (n :: Nat) (m :: Nat) = Linear.V n (Linear.V n Double)
type HMatrix = HMatrix.Matrix Double

-- | Convert a @linear@ matrix to an @hmatrix@ matrix. Super inefficient.
linear2hmatrix :: LMatrix n m -> HMatrix
linear2hmatrix =
    HMatrix.fromRows
  . Vector.toList
  . fmap (Vector.Generic.convert . Linear.toVector)
  . Linear.toVector

-- | Convert an @hmatrix@ matrix to a @linear@ matrix. Super inefficient.
hmatrix2linear :: HMatrix -> LMatrix n m
hmatrix2linear =
    Linear.V
  . Vector.fromList
  . map (Linear.V . Vector.Generic.convert)
  . HMatrix.toRows

pinv :: LMatrix n n -> LMatrix n n
pinv = hmatrix2linear . HMatrix.pinv . linear2hmatrix
