using Doji.AI.Tokenizers;
using System.Collections.Generic;
using System.Linq;

namespace Doji.AI.Transformers {

    public abstract partial class PreTrainedTokenizerBase {

        /// <summary>The named special tokens, as the tokenizer config sets them.</summary>
        protected Dictionary<string, AddedToken> SpecialTokensMap { get; } = new();

        /// <summary>The special tokens beyond the named ones (transformers' <c>extra_special_tokens</c>).</summary>
        public List<AddedToken> ExtraSpecialTokens { get; } = new();

        /// <summary>
        /// Reads the special tokens of the config, with the defaults of the tokenizer class for the ones the config does
        /// not set.
        /// </summary>
        protected void SetSpecialTokens(string unk = null, string bos = null, string eos = null, string pad = null) {
            static AddedToken Default(string token) => token != null ? new AddedToken(token, special: true) : null;
            SpecialTokensMap["bos_token"] = Config.BosToken ?? Default(bos);
            SpecialTokensMap["eos_token"] = Config.EosToken ?? Default(eos);
            SpecialTokensMap["unk_token"] = Config.UnkToken ?? Default(unk);
            SpecialTokensMap["sep_token"] = Config.SepToken;
            SpecialTokensMap["pad_token"] = Config.PadToken ?? Default(pad);
            SpecialTokensMap["cls_token"] = Config.ClsToken;
            SpecialTokensMap["mask_token"] = Config.MaskToken;
            ExtraSpecialTokens.Clear();
            ExtraSpecialTokens.AddRange(Config.ExtraSpecialTokens);
        }

        private string Special(string name) => SpecialTokensMap.TryGetValue(name, out AddedToken token) ? token?.Content : null;
        private int? SpecialId(string name) => Special(name) is string token ? ConvertTokensToIds(token) : null;

        public string BosToken => Special("bos_token");
        public string EosToken => Special("eos_token");
        public string UnkToken => Special("unk_token");
        public string SepToken => Special("sep_token");
        public string PadToken => Special("pad_token");
        public string ClsToken => Special("cls_token");
        public string MaskToken => Special("mask_token");

        public int? BosTokenId => SpecialId("bos_token");
        public int? EosTokenId => SpecialId("eos_token");
        public int? UnkTokenId => SpecialId("unk_token");
        public int? SepTokenId => SpecialId("sep_token");
        public int? PadTokenId => SpecialId("pad_token");
        public int? ClsTokenId => SpecialId("cls_token");
        public int? MaskTokenId => SpecialId("mask_token");

        /// <summary>The type id of padding tokens.</summary>
        public int PadTokenTypeId { get; set; }

        /// <summary>The named and the extra special tokens, without duplicates.</summary>
        public List<string> AllSpecialTokens {
            get {
                return SpecialTokensMap.Values.Concat(ExtraSpecialTokens).Where(t => t != null).Select(t => t.Content).Distinct().ToList();
            }
        }

        /// <summary>The ids of <see cref="AllSpecialTokens"/>.</summary>
        public List<int> AllSpecialIds => AllSpecialTokens.Select(ConvertTokensToIds).Where(id => id.HasValue).Select(id => id.Value).ToList();
    }
}
