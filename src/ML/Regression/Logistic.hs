{-# LANGUAGE ScopedTypeVariables #-}

module ML.Regression.Logistic where

import Linear.Matrix
import Linear.Vector

hypothesis
  :: (Floating a, Functor t, Foldable f, Additive f)
  => t (f a) -> f a -> t a
hypothesis xs ts = (\x -> 1 / exp (-x)) <$> (xs !* ts)
